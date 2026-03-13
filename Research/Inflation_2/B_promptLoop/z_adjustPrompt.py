import os
import concurrent.futures
import openai
from dotenv import load_dotenv
from tqdm import tqdm

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, '..', '..', '..'))

load_dotenv(os.path.join(ROOT_FOLDER, '.env'))

client = openai.OpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1",
)

MODEL                = "x-ai/grok-4.1-fast"
MAX_TOKENS           = 1024
WORD_COUNT_TOLERANCE = 10
MAX_RETRIES          = 5
N_CANDIDATES         = 10

ADJUSTER_SYSTEM = (
    "You are a prompt engineer. Rewrite the ALTERABLE PART of an evaluation prompt "
    "to shift its average bias (O=+1, N=0, P=−1) toward the target. "
    "Keep the word count within {tolerance} words of the original ({original_count} words). "
    "Return ONLY the rewritten alterable part, no explanation."
)

ADJUSTER_TEMPLATE = (
    "ALTERABLE PART (rewrite this):\n{alterable_part}\n\n"
    "FIXED PART (do not change):\n{fixed_part}\n\n"
    "Current bias: {current_bias:+.4f} > Target: {target_bias:+.4f}  ({direction})\n"
    "Return ONLY the new alterable part."
)

DIRECTION_INCREASE = "bias too low - loosen P criteria and/or tighten O criteria to shift borderline sentences toward N or O"
DIRECTION_DECREASE = "bias too high - loosen O criteria and/or tighten P criteria to shift borderline sentences toward N or P"


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _generate_one_candidate(system_message, user_message, original_word_count, word_count_tolerance):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=MAX_TOKENS,
            )
            result = response.choices[0].message.content.strip()
            result_word_count = len(result.split())
            if abs(result_word_count - original_word_count) <= word_count_tolerance:
                return result
            tqdm.write(
                f"  Word count {result_word_count} vs original {original_word_count} "
                f"(diff: {abs(result_word_count - original_word_count)}, tolerance: {word_count_tolerance}). Retrying..."
            )
        except Exception as e:
            tqdm.write(f"  Error on attempt {attempt}/{MAX_RETRIES}: {e}")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def adjust_prompt(fixed_part, alterable_part, target_bias, current_bias, original_word_count,
                  word_count_tolerance=WORD_COUNT_TOLERANCE, n_candidates=N_CANDIDATES):
    system_message = ADJUSTER_SYSTEM.format(
        tolerance=word_count_tolerance,
        original_count=original_word_count,
    )
    user_message = ADJUSTER_TEMPLATE.format(
        fixed_part=fixed_part,
        alterable_part=alterable_part,
        current_bias=current_bias,
        target_bias=target_bias,
        direction=DIRECTION_INCREASE if current_bias < target_bias else DIRECTION_DECREASE,
    )
    candidates = []
    with tqdm(total=n_candidates, desc="Generating candidates", unit="candidate", leave=False) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_candidates) as executor:
            futures = [
                executor.submit(
                    _generate_one_candidate,
                    system_message, user_message, original_word_count, word_count_tolerance,
                )
                for _ in range(n_candidates)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    candidates.append(result)
                pbar.update(1)
    if not candidates:
        tqdm.write("  All candidates failed; using original alterable part.")
        return [alterable_part]
    return candidates