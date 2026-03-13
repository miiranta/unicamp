import os
import csv
import time
import threading
import concurrent.futures
from collections import Counter
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

MODEL        = "qwen/qwen3-235b-a22b-2507"
MAX_TOKENS   = 1024
MAX_RETRIES  = 5
RETRY_SLEEP  = 2.0
MAX_WORKERS  = 10

GRADE_MAP = {"O": 1, "N": 0, "P": -1}

_print_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path):
    if not os.path.exists(path):
        return {}
    completed = {}
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter='|')
        next(reader, None)
        for row in reader:
            if len(row) == 2:
                grade_str, sentence = row
                try:
                    completed[sentence] = int(grade_str) if grade_str not in ('', 'None') else None
                except ValueError:
                    completed[sentence] = None
    return completed

def _save_checkpoint(path, grade, sentence):
    write_header = not os.path.exists(path)
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        if write_header:
            writer.writerow(['Grade', 'Sentence'])
        writer.writerow([grade, sentence])


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _parse_grade(raw):
    cleaned = raw.upper().replace('\n', '').replace('.', '').replace('<｜BEGIN▁OF▁SENTENCE｜>', '').strip()
    return GRADE_MAP.get(cleaned)

def _grade_sentence(prompt_text, sentence):
    retries_log = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt_text + sentence}],
                max_tokens=MAX_TOKENS,
            )
            grade = _parse_grade(response.choices[0].message.content)
            if grade is not None:
                return grade, retries_log
            retries_log.append(f"  Unexpected response '{response.choices[0].message.content.strip()}', retrying ({attempt}/{MAX_RETRIES})...")
        except Exception as e:
            retries_log.append(f"  Error on attempt {attempt}/{MAX_RETRIES}: {e}")
        time.sleep(RETRY_SLEEP)
    return None, retries_log


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _worker(prompt_text, sentence, checkpoint_path, checkpoint_lock):
    grade, retries_log = _grade_sentence(prompt_text, sentence)
    with _print_lock:
        for msg in retries_log:
            tqdm.write(msg)
    if checkpoint_path is not None:
        with checkpoint_lock:
            _save_checkpoint(checkpoint_path, grade, sentence)
    return grade


def test_prompt(prompt_text, sentences, checkpoint_dir=None, desc="Evaluating"):
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    total = len(sentences)
    unique_paths = {os.path.join(checkpoint_dir, f'{date}.csv') for date, _ in sentences} if checkpoint_dir else set()
    completed_by_path = {path: _load_checkpoint(path) for path in unique_paths}
    locks_by_path     = {path: threading.Lock()       for path in unique_paths}

    results_map = {}
    futures_map = {}

    with tqdm(total=total, desc=desc, unit='sent') as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for idx, (date, sentence) in enumerate(sentences, start=1):
                checkpoint_path = os.path.join(checkpoint_dir, f'{date}.csv') if checkpoint_dir else None
                completed = completed_by_path.get(checkpoint_path, {})

                if sentence in completed:
                    results_map[idx] = {'date': date, 'sentence': sentence, 'grade': completed[sentence]}
                    pbar.update(1)
                else:
                    lock = locks_by_path.get(checkpoint_path, threading.Lock())
                    future = executor.submit(
                        _worker,
                        prompt_text, sentence, checkpoint_path, lock,
                    )
                    futures_map[future] = (idx, date, sentence)

            for future in concurrent.futures.as_completed(futures_map):
                idx, date, sentence = futures_map[future]
                grade = future.result()
                results_map[idx] = {'date': date, 'sentence': sentence, 'grade': grade}
                pbar.update(1)

    results = [results_map[i] for i in range(1, total + 1)]
    valid_grades = [r['grade'] for r in results if r['grade'] is not None]
    bias = sum(valid_grades) / len(valid_grades) if valid_grades else None

    if valid_grades:
        v = len(valid_grades)
        c = Counter(valid_grades)
        print(f"  O: {c[1]} ({c[1]/v:.0%})  N: {c[0]} ({c[0]/v:.0%})  P: {c[-1]} ({c[-1]/v:.0%})  |  bias: {bias:+.4f}  ({v}/{total} valid)")

    return {
        'results': results,
        'bias': bias,
        'valid_count': len(valid_grades),
        'total_count': total,
    }