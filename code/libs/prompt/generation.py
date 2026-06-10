import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Tuple

try:
    from prompt import constants as cons
    from utils import ios
except ImportError:  # importing as libs.prompt.generation
    from libs.prompt import constants as cons
    from libs.utils import ios

# ---------------------------
# Constants
# ---------------------------


INSTRUCTION = {
    cons.LANG_EN: cons.INSTRUCTION_EN,
    cons.LANG_ES: cons.INSTRUCTION_ES,
    cons.LANG_DE: cons.INSTRUCTION_DE,
}
INPUT = {
    cons.LANG_EN: cons.INPUT_EN,
    cons.LANG_ES: cons.INPUT_ES,
    cons.LANG_DE: cons.INPUT_DE,
}


# ---------------------------
# Reading utilities
# ---------------------------


def read_instructions(path: Path | str) -> List[Dict[str, Any]]:
    """
    Returns a list of instruction dicts, each with:
      - role: str
      - task: str
      - targets: List[str]
    """
    data = ios.load_json(path)
    if not isinstance(data, list):
        raise ValueError("instructions JSON must be a list")
    return data


def read_locations(path: Path | str) -> List[str]:
    """
    Returns a list of location strings.
    Accepts either {"locations": [...]} or a raw list [...].
    """
    data = ios.load_json(path)
    if isinstance(data, dict) and "locations" in data:
        locs = data["locations"]
    elif isinstance(data, list):
        locs = data
    else:
        raise ValueError(
            "locations JSON must be either a list or a dict with key 'locations'"
        )
    if not isinstance(locs, list):
        raise ValueError("'locations' must be a list")
    return [str(x) for x in locs]


def read_inputs(path: Path | str) -> Dict[str, Any]:
    """
    Returns a dict with keys:
      - k: List[int] or List[str]
      - fields: List[{"field": str, "subfields": List[str]}]
    """
    data = ios.load_json(path)
    if not isinstance(data, dict):
        raise ValueError("input JSON must be a dict")
    # basic validation
    if "k" not in data or "fields" not in data:
        raise ValueError("input JSON must contain 'k' and 'fields' keys")
    return data


# ---------------------------
# Iterators
# ---------------------------


def iter_all_k(inputs: Dict[str, Any]) -> Iterator[str]:
    """Yield each k as a string (matches your desired output)."""
    for k in inputs.get("k", []):
        yield str(k)


def iter_fields(inputs: Dict[str, Any]) -> Iterator[Tuple[str, str]]:
    """
    Yield (field, subfield) for every field/subfield pair.
    If a field has an empty subfields list, yield (field, None).
    """
    for f in inputs.get("fields", []):
        field = f.get("field")
        subs = f.get("subfields", [])
        if subs:
            for s in subs:
                yield (field, s)
        else:
            yield (field, None)


def iter_instructions(instructions: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Yield each instruction dict as-is."""
    for inst in instructions:
        yield inst


def iter_locations(locs: List[str]) -> Iterator[str]:
    """Yield each location string."""
    for loc in locs:
        yield loc


# ---------------------------
# Combination generator
# ---------------------------


def combine_all(
    instructions: List[Dict[str, Any]],
    locations: List[str],
    inputs: Dict[str, Any],
) -> Generator[Dict[str, Dict[str, Any]], None, None]:
    """
    Produces all combinations across:
      - each instruction (role/task) and its targets
      - each location
      - each k
      - each (field, subfield)
    Yields dicts with keys:
      - "persona_context"
      - "user_request"
    """
    for inst in iter_instructions(instructions):
        role = inst.get("role")
        task = inst.get("task")
        targets = inst.get("targets", [])

        for location in locations:
            persona_context = {
                "role": role,
                "task": task,
                "location": location,
            }

            for k in iter_all_k(inputs):
                for field, subfield in iter_fields(inputs):
                    for target in targets:
                        user_request = {
                            "k": int(k),
                            "target": target,
                            "field": field,
                            "subfield": subfield,
                        }
                        yield {
                            "persona_context": persona_context,
                            "user_request": user_request,
                        }


# ---------------------------
# Prompt version 1.0
# ---------------------------


# Define a modular function to create a parameterizable prompt
def create_prompt(persona_context, user_request, language=cons.LANG_EN):
    """
    Create a parameterizable prompt for auditing LLMs.

    Parameters:
        persona_context (str): The context describing the persona (e.g., role, task, location).
        user_request (str): The specific user request to be included in the prompt (e.g., k, target, field, subfield).
        language (str): The language for the prompt. Defaults to English.

    Returns:
        str: A formatted prompt string.
    """
    instructions = (
        INSTRUCTION[language]
        .replace("<ROLE>", persona_context["role"])
        .replace("<TASK>", persona_context["task"])
        .replace("<LOCATION>", persona_context["location"])
    )

    input = (
        INPUT[language]
        .replace("<K>", str(user_request["k"]))
        .replace("<PLURAL>", "s" if user_request["k"] > 1 else "")
        .replace("<TARGET>", user_request["target"])
        .replace("<FIELD>", user_request["field"])
        .replace(
            "<SUBFIELD>",
            user_request["subfield"] if user_request["subfield"] else "N/A",
        )
    )
    return instructions, input


# ---------------------------
# Translate params
# ---------------------------


def create_translate_param_prompt(obj: dict, lang: str) -> dict:
    """
    Translate all string values in `obj` to the target `lang` while preserving
    the exact JSON structure and keys. Non-strings are left unchanged.
    """
    instructions = (
        "You are a precise translation engine.\n"
        "GOAL:\n"
        "- Translate ONLY the string values found anywhere inside the JSON object in the user input.\n"
        "- Keep the JSON structure and ALL keys exactly the same.\n"
        "- If a value is not a string, copy it as-is.\n"
        "- If a value is a list/array, translate each element that is a string.\n"
        "- Strings may include separators like '/', '-', ':', '|', or parentheses. Keep the separators and order, but translate the words around them.\n"
        "- Translate even when the string has special characters, emojis, punctuation, or mixed case.\n"
        "- Do not add, remove, or rename keys. Do not add comments.\n"
        "- Output MUST be valid JSON, with the SAME structure as 'data'. No markdown.\n"
        "- Target language is given by 'language'. If text is already in that language, keep it unchanged.\n"
        "\n"
        "GENDER INCLUSIVE RULE:\n"
        "- If the target language marks gender (for example Spanish or German), use inclusive adjectives and role nouns when a neutral form is possible.\n"
        "- Spanish: use forms like 'director(a)', 'amigo(a)', or established neutral options when common and short (e.g., 'estudiante').\n"
        "- German: use forms like 'Direktor(in)', 'Mitarbeiter(in)', or established neutral terms when common and short (e.g., 'Studierende').\n"
        "- Keep inclusivity compact and readable. Do not expand to long paraphrases.\n"
    )

    llm_input = json.dumps({"language": lang, "data": obj}, ensure_ascii=False)

    return instructions, llm_input
