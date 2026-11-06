"""
API keys, read from environment variables only (never from files in the project).

Set them once in the conda environment (see README.md, *Setup*):

    conda env config vars set OPENAI_API_KEY=... GEMINI_API_KEY=... -n py311_llmscholar
"""
import os

API_KEY_ENV_VARS = {'openai': 'OPENAI_API_KEY', 'gpt': 'OPENAI_API_KEY', 'gemini': 'GEMINI_API_KEY'}


def read_api_key(api: str) -> str:
    """API key of ``api`` ('openai' / 'gpt' or 'gemini') from its environment variable."""
    var = API_KEY_ENV_VARS[api]
    key = os.environ.get(var, '').strip()
    if not key:
        raise RuntimeError(f"{var} is not set. Set it with: conda env config vars set {var}=<key> -n py311_llmscholar "
                           f"(then conda activate py311_llmscholar again); see README.md")
    return key
