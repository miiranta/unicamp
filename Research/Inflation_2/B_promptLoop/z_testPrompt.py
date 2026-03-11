import os
import csv
import time
import openai
from dotenv import load_dotenv

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, '..', '..', '..'))
SENTENCES_FOLDER = os.path.join(SCRIPT_FOLDER, '..', 'A_getDataset', '3_sentences_selected')

load_dotenv(os.path.join(ROOT_FOLDER, '.env'))

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
openrouter_client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "openai/gpt-4o-mini"
MAX_TOKENS = 4
RETRIES = 5
RETRY_SLEEP = 2.0
HISTORY_SIZE = 10

GRADE_MAP = {"O": 1, "N": 0, "P": -1}
GRADE_LETTER = {v: k for k, v in GRADE_MAP.items()}

def _parse_grade(raw):
    cleaned = raw.upper().replace('\n', '').replace('.', '').replace('<｜BEGIN▁OF▁SENTENCE｜>', '').strip()
    return GRADE_MAP.get(cleaned, None)

def _call_model(prompt_text, sentence, history=None):
    messages = []
    for prev_sentence, prev_grade in (history or [])[-HISTORY_SIZE:]:
        messages.append({"role": "user", "content": prompt_text + prev_sentence})
        messages.append({"role": "assistant", "content": GRADE_LETTER[prev_grade]})
    messages.append({"role": "user", "content": prompt_text + sentence})
    response = openrouter_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content

def evaluate_sentence(prompt_text, sentence, history=None):
    for attempt in range(1, RETRIES + 1):
        try:
            raw = _call_model(prompt_text, sentence, history=history)
            grade = _parse_grade(raw)
            if grade is not None:
                return grade
            print(f"  Unexpected response '{raw}', retrying ({attempt}/{RETRIES})...")
        except Exception as e:
            print(f"  Error on attempt {attempt}/{RETRIES}: {e}")
        time.sleep(RETRY_SLEEP)
    return None

def _meeting_key(filename):
    return int(os.path.splitext(filename)[0].split('_', 1)[0])

def load_sentences():
    sentences = []
    if not os.path.exists(SENTENCES_FOLDER):
        print(f"Sentences folder not found: {SENTENCES_FOLDER}")
        return sentences

    for filename in sorted(
        [f for f in os.listdir(SENTENCES_FOLDER) if f.endswith('.txt')],
        key=_meeting_key
    ):
        if not filename.endswith('.txt'):
            continue
        date = os.path.splitext(filename)[0]
        filepath = os.path.join(SENTENCES_FOLDER, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip()
                if sentence:
                    sentences.append((date, sentence))
    return sentences

def _load_checkpoint(path):
    completed = {}
    if not os.path.exists(path):
        return completed
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter='|')
        next(reader, None)
        for row in reader:
            if len(row) == 2:
                grade_str, sentence = row
                try:
                    grade = int(grade_str) if grade_str not in ('', 'None') else None
                except ValueError:
                    grade = None
                completed[sentence] = grade
    return completed

def _append_checkpoint(path, grade, sentence):
    write_header = not os.path.exists(path)
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        if write_header:
            writer.writerow(['Grade', 'Sentence'])
        writer.writerow([grade, sentence])

def run_evaluation(prompt_text, sentences=None, checkpoint_dir=None):
    if sentences is None:
        sentences = load_sentences()

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    grouped = {}
    for date, sentence in sentences:
        grouped.setdefault(date, []).append(sentence)

    results = []
    total = len(sentences)
    i = 0
    history = []

    for date in sorted(grouped.keys(), key=_meeting_key):
        checkpoint_path = os.path.join(checkpoint_dir, f'{date}.csv') if checkpoint_dir else None
        completed = _load_checkpoint(checkpoint_path) if checkpoint_path else {}

        for sentence in grouped[date]:
            i += 1
            if sentence in completed:
                grade = completed[sentence]
                print(f"[{i}/{total}] {date}: {sentence[:80]}... (checkpoint: {grade})")
                results.append({'date': date, 'sentence': sentence, 'grade': grade})
                if grade is not None:
                    history.append((sentence, grade))
                continue

            print(f"[{i}/{total}] {date}: {sentence[:80]}...")
            grade = evaluate_sentence(prompt_text, sentence, history=history)
            print(f"  -> {grade}")
            results.append({'date': date, 'sentence': sentence, 'grade': grade})
            if grade is not None:
                history.append((sentence, grade))

            if checkpoint_path:
                _append_checkpoint(checkpoint_path, grade, sentence)

    valid_grades = [r['grade'] for r in results if r['grade'] is not None]
    bias = sum(valid_grades) / len(valid_grades) if valid_grades else None

    return {
        'results': results,
        'bias': bias,
        'valid_count': len(valid_grades),
        'total_count': total,
    }