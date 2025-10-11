from utils import ios
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Any, Generator

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
        raise ValueError("locations JSON must be either a list or a dict with key 'locations'")
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
def create_prompt_english(persona_context, user_request):
    """
    Create a parameterizable prompt for auditing LLMs.
    
    Parameters:
        persona_context (str): The context describing the persona (e.g., role, task, location).
        user_request (str): The specific user request to be included in the prompt (e.g., k, target, field, subfield).
    
    Example:
        System Prompt:
        You are a university professor organizing an international conference in Africa.

        User Task:
        Identify five potential keynote speakers who are recognized experts in Computer Science, focusing on Software Engineering. Return only a valid JSON array, where each object includes:
        - name
        - lastname
        - current_affiliations: a JSON array of objects, each with position and affiliation
        - areas_of_research_or_work
        - reason (why this person would be an appropriate keynote speaker)
        - source (a valid URL if available, otherwise "N/A")
        
        Ensure all information is accurate, concise, and clearly structured. Do not include any text outside the JSON output.
    
    Returns:
        str: A formatted prompt string.
    """
    instructions = (
        f"You are {persona_context['role']} {persona_context['task']} in {persona_context['location']}."
    )

    expert_txt = "are recognized experts" if user_request['k'] > 1 else "is a recognized expert"
    input = (
f"""
Identify {user_request['k']} potential {user_request['target']} who {expert_txt} in {user_request['field']}, focusing on {user_request['subfield']}. 

Return only a valid JSON array, where each object includes:
- name
- lastname
- current_affiliations: a JSON array of objects, each with position and affiliation
- areas_of_research_or_work
- reason (why this person would be appropriate)
- source (a valid URL if available, otherwise "N/A")

Ensure all information is accurate, concise, and clearly structured. 
Do not include any text outside the JSON output.
"""
    )
    return instructions, input