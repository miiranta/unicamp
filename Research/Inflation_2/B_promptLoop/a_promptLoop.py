import os
import json
import csv
import random
import shutil

from z_testPrompt import test_prompt
from z_adjustPrompt import adjust_prompt

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
ITERATIONS_FOLDER = os.path.join(SCRIPT_FOLDER, 'iterations')
SENTENCES_FOLDER  = os.path.join(SCRIPT_FOLDER, '..', 'A_getDataset', '3_sentences_selected')

TARGET_BIAS     = 0.0
TOLERANCE       = 0.0025
MAX_ITERATIONS  = 50
SAMPLE_FRACTION = 0.05

PROMPT_ALTERABLE = """
DEFINIÇÃO DE OTIMISMO:
Ocorre quando as projeções indicam que a inflação ficará abaixo da meta ou dentro do intervalo de tolerância com folga. 
Isso pode sinalizar que o Banco Central vê espaço para reduzir juros ou manter uma política monetária mais acomodatícia. 

DEFINIÇÃO DE PESSIMISMO:
Ocorre quando as projeções apontam para inflação acima da meta ou próxima do teto do intervalo de tolerância. 
Isso sugere preocupação com pressões inflacionárias e pode justificar uma política monetária mais restritiva.
"""

PROMPT_FIXED = """

AVALIE A FRASE COMO
O para OTIMISTA
N para NEUTRA
P para PESSIMISTA
SUA RESPOSTA DEVE SER APENAS UMA LETRA, SEM QUALQUER OUTRO TEXTO

A FRASE É:
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _meeting_key(filename):
    return int(os.path.splitext(filename)[0].split('_', 1)[0])


def load_sentences():
    sentences = []
    if not os.path.exists(SENTENCES_FOLDER):
        print(f"Sentences folder not found: {SENTENCES_FOLDER}")
        return sentences
    for filename in sorted(
        [f for f in os.listdir(SENTENCES_FOLDER) if f.endswith('.txt')],
        key=_meeting_key,
    ):
        date = os.path.splitext(filename)[0]
        with open(os.path.join(SENTENCES_FOLDER, filename), 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip()
                if sentence:
                    sentences.append((date, sentence))
    return sentences


# ---------------------------------------------------------------------------
# Iteration management
# ---------------------------------------------------------------------------

def _iteration_folder(n):
    path = os.path.join(ITERATIONS_FOLDER, f'iteration_{n}')
    os.makedirs(path, exist_ok=True)
    return path


def _load_state():
    if not os.path.exists(ITERATIONS_FOLDER):
        return 1, PROMPT_ALTERABLE, float('inf'), PROMPT_ALTERABLE

    iter_nums = sorted([
        int(d.split('_', 1)[1])
        for d in os.listdir(ITERATIONS_FOLDER)
        if os.path.isdir(os.path.join(ITERATIONS_FOLDER, d))
        and d.startswith('iteration_') and d.split('_', 1)[1].isdigit()
    ])

    if not iter_nums:
        return 1, PROMPT_ALTERABLE, float('inf'), PROMPT_ALTERABLE

    start_iteration = iter_nums[-1]
    current_prompt = PROMPT_ALTERABLE
    best_error = float('inf')
    best_prompt = PROMPT_ALTERABLE

    for n in iter_nums:
        summary_path = os.path.join(ITERATIONS_FOLDER, f'iteration_{n}', 'summary.json')
        if not os.path.exists(summary_path):
            continue
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        prompt = data.get('prompt_alterable', PROMPT_ALTERABLE)
        if 'bias' in data:
            error = abs(data['bias'] - TARGET_BIAS)
            if error < best_error:
                best_error = error
                best_prompt = prompt
            if n == iter_nums[-1]:
                start_iteration = n + 1
                current_prompt = prompt
        else:
            if n == iter_nums[-1]:
                start_iteration = n
                current_prompt = prompt

    return start_iteration, current_prompt, best_error, best_prompt


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _save_summary(folder, iteration_n, prompt_alterable, evaluation=None):
    summary = {
        'iteration': iteration_n,
        'prompt_alterable': prompt_alterable,
        'prompt_fixed': PROMPT_FIXED,
    }
    if evaluation is not None:
        summary.update({
            'bias': evaluation['bias'],
            'valid_count': evaluation['valid_count'],
            'total_count': evaluation['total_count'],
            'target_bias': TARGET_BIAS,
            'tolerance': TOLERANCE,
        })
        with open(os.path.join(folder, 'results.csv'), 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(['Date', 'Grade', 'Sentence'])
            for row in evaluation['results']:
                writer.writerow([row['date'], row['grade'], row['sentence']])
        checkpoints_dir = os.path.join(folder, 'checkpoints')
        if os.path.exists(checkpoints_dir):
            shutil.rmtree(checkpoints_dir)
    with open(os.path.join(folder, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def _select_best_candidate(candidates, sentences, sample_size):
    print(f"\n  Pre-screening {len(candidates)} candidates on {sample_size} sentences...")
    sample               = random.sample(sentences, sample_size)
    best_candidate       = candidates[0]
    best_candidate_error = float('inf')
    for i, candidate in enumerate(candidates):
        candidate_prompt = candidate + "\n\n" + PROMPT_FIXED + "\n"
        sample_eval      = test_prompt(candidate_prompt, sentences=sample, desc="Pre-screening")
        sample_bias      = sample_eval['bias']
        if sample_bias is None:
            print(f"  Candidate {i+1}: no valid evaluations, skipping.")
            continue
        candidate_error = abs(sample_bias - TARGET_BIAS)
        print(f"  Candidate {i+1}: bias {sample_bias:+.4f}  error: {candidate_error:.4f}")
        if candidate_error < best_candidate_error:
            best_candidate_error = candidate_error
            best_candidate       = candidate
    print(f"  \u2192 selected candidate  error: {best_candidate_error:.4f}")
    return best_candidate


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    os.makedirs(ITERATIONS_FOLDER, exist_ok=True)

    sentences = load_sentences()
    if not sentences:
        print("No sentences found. Exiting.")
        return

    start_iteration, prompt_alterable, best_error, best_prompt_alterable = _load_state()
    original_word_count = len(PROMPT_ALTERABLE.split())

    sample_size = max(1, int(len(sentences) * SAMPLE_FRACTION))
    print(f"Sentences: {len(sentences)}  |  sample: {sample_size}  |  target: {TARGET_BIAS} ± {TOLERANCE}  |  max iter: {MAX_ITERATIONS}")
    if start_iteration > 1:
        print(f"Resuming from iteration {start_iteration}  |  best known error: {best_error:.4f}")
    print()

    for iteration in range(start_iteration, start_iteration + MAX_ITERATIONS):
        folder = _iteration_folder(iteration)
        full_prompt = prompt_alterable + "\n\n" + PROMPT_FIXED + "\n"

        print(f"{'─'*60}")
        print(f"  Iteration {iteration}\n{prompt_alterable.strip()}")
        print(f"{'─'*60}")

        _save_summary(folder, iteration, prompt_alterable)
        evaluation = test_prompt(full_prompt, sentences=sentences, checkpoint_dir=os.path.join(folder, 'checkpoints'), desc="Evaluating")
        current_bias = evaluation['bias']

        if current_bias is None:
            print("No valid evaluations. Stopping.")
            break

        _save_summary(folder, iteration, prompt_alterable, evaluation)

        current_error = abs(current_bias - TARGET_BIAS)
        new_best = current_error < best_error
        if new_best:
            best_error = current_error
            best_prompt_alterable = prompt_alterable
        print(f"  → bias: {current_bias:+.4f}  (target: {TARGET_BIAS} ± {TOLERANCE})" + ("  ★ new best" if new_best else ""))

        if abs(current_bias - TARGET_BIAS) <= TOLERANCE:
            print(f"\nConverged at iteration {iteration}.")
            break

        if iteration == start_iteration + MAX_ITERATIONS - 1:
            print(f"\nMax iterations ({MAX_ITERATIONS}) reached.")
            break

        candidates = adjust_prompt(
            fixed_part=PROMPT_FIXED,
            alterable_part=best_prompt_alterable,
            target_bias=TARGET_BIAS,
            current_bias=current_bias,
            original_word_count=original_word_count,
        )

        if len(candidates) == 1:
            prompt_alterable = candidates[0]
        else:
            prompt_alterable = _select_best_candidate(candidates, sentences, sample_size)

        print(f"\n  New prompt:\n{prompt_alterable.strip()}\n")

if __name__ == "__main__":
    main()
    