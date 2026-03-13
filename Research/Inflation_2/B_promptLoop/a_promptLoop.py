import os
import json
import csv
import random
import shutil

from z_testPrompt import run_evaluation, load_sentences
from z_adjustPrompt import adjust_prompt

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
ITERATIONS_FOLDER = os.path.join(SCRIPT_FOLDER, 'iterations')

TARGET_BIAS = 0.0
TOLERANCE   = 0.025
MAX_ITERATIONS = 50
SAMPLE_FRACTION = 0.3

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

def _iteration_folder(n):
    path = os.path.join(ITERATIONS_FOLDER, f'iteration_{n}')
    os.makedirs(path, exist_ok=True)
    return path

def _resume_state():
    if not os.path.exists(ITERATIONS_FOLDER):
        return 1, PROMPT_ALTERABLE

    numbers = sorted([
        int(d.split('_', 1)[1])
        for d in os.listdir(ITERATIONS_FOLDER)
        if os.path.isdir(os.path.join(ITERATIONS_FOLDER, d))
        and d.startswith('iteration_') and d.split('_', 1)[1].isdigit()
    ])

    if not numbers:
        return 1, PROMPT_ALTERABLE

    last = numbers[-1]
    summary_path = os.path.join(ITERATIONS_FOLDER, f'iteration_{last}', 'summary.json')

    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'bias' in data:
            return last + 1, data.get('prompt_alterable', PROMPT_ALTERABLE)
        return last, data.get('prompt_alterable', PROMPT_ALTERABLE)

    return last, PROMPT_ALTERABLE

def _find_best_known():
    if not os.path.exists(ITERATIONS_FOLDER):
        return None, PROMPT_ALTERABLE
    best_error = float('inf')
    best_bias = None
    best_prompt = PROMPT_ALTERABLE
    for d in os.listdir(ITERATIONS_FOLDER):
        if not (os.path.isdir(os.path.join(ITERATIONS_FOLDER, d))
                and d.startswith('iteration_') and d.split('_', 1)[1].isdigit()):
            continue
        summary_path = os.path.join(ITERATIONS_FOLDER, d, 'summary.json')
        if not os.path.exists(summary_path):
            continue
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'bias' not in data:
            continue
        error = abs(data['bias'] - TARGET_BIAS)
        if error < best_error:
            best_error = error
            best_bias = data['bias']
            best_prompt = data.get('prompt_alterable', PROMPT_ALTERABLE)
    return best_bias, best_prompt

def _save_partial_summary(folder, iteration_n, prompt_alterable):
    summary = {
        'iteration': iteration_n,
        'prompt_alterable': prompt_alterable,
        'prompt_fixed': PROMPT_FIXED,
    }
    with open(os.path.join(folder, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _save_results(folder, iteration_n, prompt_alterable, evaluation):
    with open(os.path.join(folder, 'results.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(['Date', 'Grade', 'Sentence'])
        for row in evaluation['results']:
            writer.writerow([row['date'], row['grade'], row['sentence']])

    summary = {
        'iteration': iteration_n,
        'bias': evaluation['bias'],
        'valid_count': evaluation['valid_count'],
        'total_count': evaluation['total_count'],
        'target_bias': TARGET_BIAS,
        'tolerance': TOLERANCE,
        'prompt_alterable': prompt_alterable,
        'prompt_fixed': PROMPT_FIXED,
    }
    with open(os.path.join(folder, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    checkpoints_dir = os.path.join(folder, 'checkpoints')
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)

def main():
    os.makedirs(ITERATIONS_FOLDER, exist_ok=True)

    sentences = load_sentences()
    if not sentences:
        print("No sentences found. Exiting.")
        return

    start_iteration, prompt_alterable = _resume_state()
    original_word_count = len(PROMPT_ALTERABLE.split())
    if start_iteration > 1:
        print(f"Resuming from iteration {start_iteration}.")

    best_known_bias, best_prompt_alterable = _find_best_known()
    if best_known_bias is not None:
        best_error = abs(best_known_bias - TARGET_BIAS)
        print(f"Best known bias from history: {best_known_bias:.4f}  (error: {best_error:.4f})")
        print(f"Adjustments will start from the best-known prompt.")
    else:
        best_error = float('inf')
        best_prompt_alterable = PROMPT_ALTERABLE

    sample_size = max(1, int(len(sentences) * SAMPLE_FRACTION))
    print(f"Loaded {len(sentences)} sentences  (candidate sample size: {sample_size}).")
    print(f"Target bias: {TARGET_BIAS}  |  Tolerance: {TOLERANCE}  |  Max iterations: {MAX_ITERATIONS}\n")

    for iteration in range(start_iteration, start_iteration + MAX_ITERATIONS):
        folder = _iteration_folder(iteration)
        full_prompt = prompt_alterable + "\n\n" + PROMPT_FIXED + "\n"

        print(f"{'='*60}")
        print(f"Iteration {iteration}")
        print(f"Alterable prompt: \n{prompt_alterable}")
        print(f"{'='*60}")

        _save_partial_summary(folder, iteration, prompt_alterable)
        evaluation = run_evaluation(full_prompt, sentences=sentences, checkpoint_dir=os.path.join(folder, 'checkpoints'))
        current_bias = evaluation['bias']

        if current_bias is None:
            print("No valid evaluations returned. Stopping.")
            break

        print(f"\n  >> Bias: {current_bias:.4f}  (target: {TARGET_BIAS}, tolerance: ±{TOLERANCE})")
        _save_results(folder, iteration, prompt_alterable, evaluation)
        print(f"  Saved iteration {iteration}.")

        current_error = abs(current_bias - TARGET_BIAS)
        if current_error < best_error:
            best_error = current_error
            best_prompt_alterable = prompt_alterable
            print(f"  [NEW BEST] error {best_error:.4f}  bias {current_bias:.4f}")

        if abs(current_bias - TARGET_BIAS) <= TOLERANCE:
            print(f"\nConverged! Bias {current_bias:.4f} is within {TOLERANCE} of target {TARGET_BIAS}.")
            break

        if iteration == start_iteration + MAX_ITERATIONS - 1:
            print(f"\nReached max iterations ({MAX_ITERATIONS}). Stopping.")
            break

        print("\n  Adjusting prompt (generating candidates from best-known prompt)...")
        candidates = adjust_prompt(
            fixed_part=PROMPT_FIXED,
            alterable_part=best_prompt_alterable,
            target_bias=TARGET_BIAS,
            current_bias=current_bias,
            original_word_count=original_word_count,
        )

        if len(candidates) == 1:
            prompt_alterable = candidates[0]
            print(f"  Single candidate accepted.")
        else:
            print(f"  Pre-screening {len(candidates)} candidates on {sample_size} sampled sentences...")
            sample = random.sample(sentences, sample_size)
            best_candidate = candidates[0]
            best_candidate_error = float('inf')
            for i, candidate in enumerate(candidates):
                candidate_prompt = candidate + "\n\n" + PROMPT_FIXED + "\n"
                sample_eval = run_evaluation(candidate_prompt, sentences=sample)
                sample_bias = sample_eval['bias']
                if sample_bias is None:
                    print(f"    Candidate {i+1}: no valid sample evaluations, skipping.")
                    continue
                candidate_error = abs(sample_bias - TARGET_BIAS)
                print(f"    Candidate {i+1}: sample bias = {sample_bias:.4f}  (error: {candidate_error:.4f})")
                if candidate_error < best_candidate_error:
                    best_candidate_error = candidate_error
                    best_candidate = candidate
            prompt_alterable = best_candidate
            print(f"  Selected candidate with sample error {best_candidate_error:.4f}.")

        print(f"  New alterable part:\n{prompt_alterable}\n")

if __name__ == "__main__":
    main()
    