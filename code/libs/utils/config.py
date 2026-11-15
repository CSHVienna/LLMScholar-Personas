"""
API keys, read from environment variables only (never from files in the project).

Set them once in the conda environment (see README.md, *Setup*):

    conda env config vars set OPENAI_API_KEY=... GEMINI_API_KEY=... -n py311_llmscholar
"""
import os

from utils import constants as cons

API_KEY_ENV_VARS = {cons.SOURCE_GPT: 'OPENAI_API_KEY', cons.SOURCE_GEMINI: 'GEMINI_API_KEY'}   # api (source) -> variable


def read_api_key(api: str) -> str:
    """API key of ``api`` (cons.SOURCE_GPT 'gpt' or cons.SOURCE_GEMINI 'gemini') from its environment variable."""
    var = API_KEY_ENV_VARS[api]
    key = os.environ.get(var, '').strip()
    if not key:
        raise RuntimeError(f"{var} is not set. Set it with: conda env config vars set {var}=<key> -n py311_llmscholar "
                           f"(then conda activate py311_llmscholar again); see README.md")
    return key
