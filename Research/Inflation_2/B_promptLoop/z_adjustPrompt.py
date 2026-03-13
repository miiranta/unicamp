import os
import concurrent.futures
import openai
from dotenv import load_dotenv

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, '..', '..', '..'))

load_dotenv(os.path.join(ROOT_FOLDER, '.env'))

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
openrouter_client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "google/gemma-3-27b-it"
MAX_TOKENS = 1024
WORD_COUNT_TOLERANCE = 10
MAX_RETRIES = 5
N_CANDIDATES = 3

ADJUSTER_SYSTEM = (
    "You are an expert in NLP prompt engineering for sentiment/bias evaluation tasks. "
    "Adjust the alterable part of an evaluation prompt so that the average bias moves toward the target. "
    "Bias scale: O (optimistic) = 1, N (neutral) = 0, P (pessimistic) = -1. "
    "Do not change the fixed part. "
    "The new alterable part MUST have a word count within {tolerance} words "
    "of the original alterable part (original: {original_count} words) — do not significantly expand or shrink the text. "
    "Return ONLY the new alterable part, no explanation."
)

ADJUSTER_TEMPLATE = """
CURRENT ALTERABLE PART:
{alterable_part}

FIXED PART (do not change):
{fixed_part}

Current average bias: {current_bias:.4f} | Target bias: {target_bias:.4f}
Problem: {direction}

Rewrite the ALTERABLE PART to shift the classification boundary so that more sentences land in the
correct category. Focus on the definition wording that determines whether a sentence qualifies as
O, N, or P — tighten or loosen the threshold criteria as described above.
Return ONLY the new alterable part.
"""

DIRECTION_INCREASE = (
    "the current prompt is classifying too many sentences as P (pessimistic). "
    "Raise the bar for what qualifies as P (make the criteria stricter) "
    "and/or lower the bar for what qualifies as O (make the criteria more inclusive), "
    "so that borderline sentences shift to N or O."
)
DIRECTION_DECREASE = (
    "the current prompt is classifying too many sentences as O (optimistic). "
    "Raise the bar for what qualifies as O (make the criteria stricter) "
    "and/or lower the bar for what qualifies as P (make the criteria more inclusive), "
    "so that borderline sentences shift to N or P."
)

def _generate_one_candidate(system_message, user_message, original_word_count, word_count_tolerance, alterable_part):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = openrouter_client.chat.completions.create(
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
            print(
                f"  Candidate word count {result_word_count} is "
                f"{abs(result_word_count - original_word_count)} words from original "
                f"({original_word_count}), tolerance {word_count_tolerance}. Retrying..."
            )
        except Exception as e:
            print(f"  Error generating candidate (attempt {attempt}/{MAX_RETRIES}): {e}")
    return None

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_candidates) as executor:
        futures = [
            executor.submit(
                _generate_one_candidate,
                system_message, user_message, original_word_count, word_count_tolerance, alterable_part,
            )
            for _ in range(n_candidates)
        ]
        results = [f.result() for f in futures]
    candidates = [r for r in results if r is not None]
    if not candidates:
        print("All candidates failed word-count check; returning original alterable part.")
        return [alterable_part]
    print(f"  Generated {len(candidates)}/{n_candidates} valid candidates.")
    return candidates