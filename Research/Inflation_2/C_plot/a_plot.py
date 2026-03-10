import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
ITERATIONS_FOLDER = os.path.join(SCRIPT_FOLDER, '..', 'B_promptLoop', 'iterations')
PLOTS_FOLDER = os.path.join(SCRIPT_FOLDER, 'plots')

def _meeting_key(name):
    return int(name.split('_', 1)[1])

def load_summaries():
    if not os.path.exists(ITERATIONS_FOLDER):
        print(f"Iterations folder not found: {ITERATIONS_FOLDER}")
        return []

    dirs = sorted(
        [d for d in os.listdir(ITERATIONS_FOLDER)
         if os.path.isdir(os.path.join(ITERATIONS_FOLDER, d)) and d.startswith('iteration_') and d.split('_', 1)[1].isdigit()],
        key=_meeting_key
    )

    summaries = []
    for d in dirs:
        path = os.path.join(ITERATIONS_FOLDER, d, 'summary.json')
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'bias' in data:
            summaries.append(data)
    return summaries

def plot_bias_over_iterations(summaries):
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    iterations = [s['iteration'] for s in summaries]
    biases = [s['bias'] for s in summaries]
    target = summaries[0]['target_bias']
    tolerance = summaries[0]['tolerance']

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axhspan(target - tolerance, target + tolerance, color='green', alpha=0.1, label=f'Tolerance (±{tolerance})')
    ax.axhline(target, color='green', linestyle='--', linewidth=1.2, label=f'Target ({target})')

    ax.plot(iterations, biases, marker='o', color='steelblue', linewidth=2, label='Bias')

    for i, b in zip(iterations, biases):
        ax.annotate(f'{b:.3f}', (i, b), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Bias')
    ax.set_title('Bias per Iteration')
    ax.set_xticks(iterations)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_FOLDER, 'bias_over_iterations.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved: {out_path}')

def main():
    summaries = load_summaries()
    if not summaries:
        print('No complete summaries found.')
        return
    print(f'Found {len(summaries)} complete iterations.')
    plot_bias_over_iterations(summaries)

if __name__ == '__main__':
    main()
