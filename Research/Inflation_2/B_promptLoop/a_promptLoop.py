import os
import json
import csv
import shutil

from z_testPrompt import run_evaluation, load_sentences
from z_adjustPrompt import adjust_prompt

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
ITERATIONS_FOLDER = os.path.join(SCRIPT_FOLDER, 'iterations')

TARGET_BIAS = 0.0
TOLERANCE   = 0.05
MAX_ITERATIONS = 20

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
    if start_iteration > 1:
        print(f"Resuming from iteration {start_iteration}.")

    print(f"Loaded {len(sentences)} sentences.")
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

        if abs(current_bias - TARGET_BIAS) <= TOLERANCE:
            print(f"\nConverged! Bias {current_bias:.4f} is within {TOLERANCE} of target {TARGET_BIAS}.")
            break

        if iteration == start_iteration + MAX_ITERATIONS - 1:
            print(f"\nReached max iterations ({MAX_ITERATIONS}). Stopping.")
            break

        print("\n  Adjusting prompt...")
        prompt_alterable = adjust_prompt(
            fixed_part=PROMPT_FIXED,
            alterable_part=prompt_alterable,
            target_bias=TARGET_BIAS,
            current_bias=current_bias,
        )
        print(f"  New alterable part: {prompt_alterable}\n")

if __name__ == "__main__":
    main()
    