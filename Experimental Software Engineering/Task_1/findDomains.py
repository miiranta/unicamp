import os
import sys
import json
import pathlib
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

SCRIPT_DIR = pathlib.Path(__file__).parent
SOURCE_DIR = SCRIPT_DIR / "source"
OUTPUT_FILE = SCRIPT_DIR / "output.txt"

SYSTEM_PROMPT = (
    "You are an expert domain extractor working with a codebase. "
    "Your goal is to find every domain name and hostname present in the source files "
    "no matter how they appear: URLs, HTML href/src/action, JS strings, config values, "
    "CSS, comments, redirects, API endpoints, CDN references, etc.\n\n"
    "You have four tools:\n"
    "  list_files     - see all available files with sizes and chunk counts\n"
    "  read_chunk     - read a specific chunk of a file (you pick which ones)\n"
    "  search_text    - search for a text string across all files; returns file paths and chunk indexes of every match\n"
    "  report_domains - report domains found so far (call multiple times as you go)\n\n"
    "Strategy: call list_files first, then use search_text to quickly locate chunks "
    "that contain domain-like patterns, and read_chunk for deeper inspection. After "
    "reading each file or batch, call report_domains with what you found so far. "
    "When you have read everything relevant, call report_domains one final time with is_final=true. "
    "Collect bare domains only (e.g. example.com, cdn.host.org) - strip protocols, "
    "paths, ports, and query strings."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all source files with their byte sizes and number of chunks.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_chunk",
            "description": "Read a specific chunk of a source file by 0-based chunk index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Relative file path exactly as returned by list_files",
                    },
                    "chunk_index": {
                        "type": "integer",
                        "description": "0-based index of the chunk to read",
                    },
                },
                "required": ["filename", "chunk_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search for a text string across all source files. Returns a list of matches with file path and chunk index where each match was found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The text to search for (case-insensitive)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_domains",
            "description": (
                "Report domains found so far. Call incrementally as you discover domains. "
                "Set is_final=true on the last call to signal you are done."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Bare domain names found in the chunks read so far",
                    },
                    "is_final": {
                        "type": "boolean",
                        "description": "Set to true when you have finished reading all relevant files",
                    },
                },
                "required": ["domains", "is_final"],
            },
        },
    },
]

def build_index(source_dir: pathlib.Path, chunk_size: int) -> dict:
    index = {}
    for path in sorted(source_dir.rglob("*")):
        if any(p.startswith(".") for p in path.parts[len(source_dir.parts):]):
            continue
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks = [text[i : i + chunk_size] for i in range(0, max(len(text), 1), chunk_size)]
            rel = str(path.relative_to(source_dir))
            index[rel] = {"size": len(text), "chunk_count": len(chunks), "chunks": chunks}
        except Exception:
            pass
    return index

def handle_tool(name: str, args: dict, index: dict, accumulated: set) -> tuple[str, bool]:
    if name == "list_files":
        if not index:
            return "No files found.", False
        lines = [
            f"{fname}  ({meta['size']} bytes, {meta['chunk_count']} chunk(s))"
            for fname, meta in index.items()
        ]
        return "\n".join(lines), False

    if name == "read_chunk":
        fname = args.get("filename", "")
        idx = int(args.get("chunk_index", 0))
        if fname not in index:
            return f"ERROR: '{fname}' not found. Use list_files to see valid names.", False
        chunks = index[fname]["chunks"]
        if not (0 <= idx < len(chunks)):
            return f"ERROR: chunk_index {idx} out of range (0-{len(chunks) - 1}).", False
        return chunks[idx], False

    if name == "search_text":
        query = args.get("query", "").lower()
        if not query:
            return "ERROR: 'query' must be a non-empty string.", False
        matches = []
        for fname, meta in index.items():
            for i, chunk in enumerate(meta["chunks"]):
                if query in chunk.lower():
                    matches.append(f"{fname}  chunk {i}")
        if not matches:
            return f"No matches found for '{query}'.", False
        return f"Found {len(matches)} match(es):\n" + "\n".join(matches), False

    if name == "report_domains":
        new_domains = {d.strip().lower() for d in args.get("domains", []) if d.strip()}
        accumulated.update(new_domains)
        is_final = bool(args.get("is_final", False))
        OUTPUT_FILE.write_text("\n".join(sorted(accumulated)) + "\n", encoding="utf-8")
        return f"Recorded {len(new_domains)} new domain(s). Total so far: {len(accumulated)}.", is_final

    return "ERROR: unknown tool.", False

def assistant_msg(msg) -> dict:
    d: dict = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d

def main() -> None:
    load_dotenv(find_dotenv(usecwd=False, raise_error_if_not_found=False))

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set. Add it to a .env file.")
        sys.exit(1)

    if not SOURCE_DIR.exists():
        SOURCE_DIR.mkdir(parents=True)
        print(f"WARNING: '{SOURCE_DIR}' did not exist - it has been created.")
        print("Add your source files there and run the script again.")
        sys.exit(0)

    model = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.5")
    chunk_size = int(os.getenv("CHUNK_SIZE", "8000"))

    index = build_index(SOURCE_DIR, chunk_size)
    if not index:
        print(f"WARNING: No readable files found in '{SOURCE_DIR}'.")
        sys.exit(0)

    print(f"Model  : {model}")
    print(f"Files  : {len(index)} indexed")
    print(f"Chunks : {sum(m['chunk_count'] for m in index.values())} total\n")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Please find all domain names in the source files."},
    ]

    accumulated: set[str] = set()
    step = 0

    while True:
        step += 1
        print(f"[step {step}] Querying LLM ...", flush=True)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message
        messages.append(assistant_msg(msg))

        if not msg.tool_calls:
            print("  (no tool call - re-prompting)")
            messages.append({"role": "user", "content": "Please call report_domains with is_final=true when done."})
            continue

        done = False
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            if name == "report_domains":
                print(f"  -> report_domains ({len(args.get('domains', []))} domain(s), final={args.get('is_final', False)})")
            else:
                arg_summary = ", ".join(f"{k}={v!r}" for k, v in args.items())
                print(f"  -> {name}({arg_summary})")

            result, is_final = handle_tool(name, args, index, accumulated)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            if is_final:
                done = True

        if done:
            break

    sorted_domains = sorted(accumulated)
    OUTPUT_FILE.write_text("\n".join(sorted_domains) + "\n", encoding="utf-8")
    print(f"\nDone - {len(sorted_domains)} unique domain(s) written to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
