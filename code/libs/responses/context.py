"""
Canonicalisation of prompt parameters.

Spanish and German prompts were generated from translated parameter files
(``data/context/<language>/*.json``). Each translated file has exactly the same
structure as the English one, so a translated value is mapped back to its
canonical (English) value by its *position* in that structure.
"""

from pathlib import Path
from typing import Dict, Iterable

from utils import constants as cons
from utils import ios


class PromptContext:
    """Translate the prompt parameters stored in a response record into canonical values.

    Parameters
    ----------
    context_dir : path to ``data/context`` (one sub-folder per language).
    languages   : languages to load; English is always loaded as the reference.
    """

    def __init__(self, context_dir: Path | str, languages: Iterable[str] = cons.LANGUAGES):
        self.context_dir = Path(context_dir)
        self.languages = list(dict.fromkeys([cons.LANG_EN, *languages]))  # English first, no duplicates
        reference = self._flatten(cons.LANG_EN)

        # lookup[language][variable][translated value] -> canonical value
        self.lookup: Dict[str, Dict[str, Dict[str, str]]] = {
            language: self._align(self._flatten(language), reference) for language in self.languages
        }

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def canonical(self, language: str, parameters: dict) -> dict:
        """Return the canonical prompt variables of one response record.

        ``parameters`` is the ``parameters`` entry of a raw record, i.e.
        ``{"persona_context": {role, task, location}, "user_request": {k, target, field, subfield}}``.
        The ``task`` is dropped because it is fully determined by ``role``.
        """
        persona = parameters.get('persona_context', {})
        request = parameters.get('user_request', {})
        target = self._translate(language, 'target', request.get('target'))

        return {cons.VAR_LANGUAGE: language,
                cons.VAR_ROLE: self._translate(language, cons.VAR_ROLE, persona.get('role')),
                cons.VAR_LOCATION: self._translate(language, cons.VAR_LOCATION, persona.get('location')),
                cons.VAR_K: int(request.get('k')),
                cons.VAR_SENIORITY: cons.SENIORITY_BY_TARGET[target],
                cons.VAR_FIELD: self._translate(language, cons.VAR_FIELD, request.get('field')),
                cons.VAR_SUBFIELD: self._translate(language, cons.VAR_SUBFIELD, request.get('subfield'))}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _translate(self, language: str, variable: str, value: str) -> str:
        """Map one translated value to its canonical value (raises if unknown)."""
        try:
            return self.lookup[language][variable][value.strip()]
        except (KeyError, AttributeError) as error:
            raise KeyError(f"Unknown {variable}={value!r} for language={language!r}") from error

    def _flatten(self, language: str) -> Dict[str, list]:
        """Read the parameter files of one language as ordered lists per variable."""
        folder = self.context_dir / language
        instructions = ios.load_json(folder / 'instructions.json')
        locations = ios.load_json(folder / 'locations.json')['locations']
        inputs = ios.load_json(folder / 'input.json')

        return {cons.VAR_ROLE: [inst['role'] for inst in instructions],
                'target': [target for inst in instructions for target in inst['targets']],
                cons.VAR_LOCATION: locations,
                cons.VAR_FIELD: [f['field'] for f in inputs['fields']],
                cons.VAR_SUBFIELD: [s for f in inputs['fields'] for s in f['subfields']]}

    @staticmethod
    def _align(translated: Dict[str, list], reference: Dict[str, list]) -> Dict[str, Dict[str, str]]:
        """Pair translated and reference values position by position."""
        lookup = {}
        for variable, ref_values in reference.items():
            values = translated[variable]
            if len(values) != len(ref_values):
                raise ValueError(f"'{variable}' has {len(values)} values but English has {len(ref_values)}")
            lookup[variable] = {value.strip(): ref for value, ref in zip(values, ref_values)}
        return lookup
