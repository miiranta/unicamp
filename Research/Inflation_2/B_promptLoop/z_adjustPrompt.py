import os
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

MODEL = "openai/gpt-4o-mini"
MAX_TOKENS = 1024

ADJUSTER_SYSTEM = (
    "You are an expert in NLP prompt engineering for sentiment/bias evaluation tasks. "
    "Adjust the alterable part of an evaluation prompt so that the average bias moves toward the target. "
    "Bias scale: O (optimistic) = 1, N (neutral) = 0, P (pessimistic) = -1. "
    "Do not change the fixed part. Return ONLY the new alterable part, no explanation."
)

ADJUSTER_TEMPLATE = """
CURRENT ALTERABLE PART:
{alterable_part}

FIXED PART (do not change):
{fixed_part}

Current bias: {current_bias:.4f} | Target bias: {target_bias:.4f} | Need to {direction}.

Rewrite the ALTERABLE PART with subtle, incremental changes to move the bias toward the target.
Return ONLY the new alterable part.
"""

def adjust_prompt(fixed_part, alterable_part, target_bias, current_bias):
    user_message = ADJUSTER_TEMPLATE.format(
        fixed_part=fixed_part,
        alterable_part=alterable_part,
        current_bias=current_bias,
        target_bias=target_bias,
        direction="increase bias" if current_bias < target_bias else "decrease bias",
    )
    try:
        response = openrouter_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": ADJUSTER_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error adjusting prompt: {e}")
        return alterable_part