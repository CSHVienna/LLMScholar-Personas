"""
Prompt field -> OpenAlex mapping (data/mappings/field_mapping.json).

An author belongs to a prompt field if they have >= 1 work (and, optionally, a minimum share, A.4) whose PRIMARY topic lies in
one of the mapped OpenAlex fields or subfields. Subfields of the prompts are not mapped.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from utils import constants as cons
from utils import ios


@dataclass(frozen=True)
class FieldMapping:
    field: str                    # prompt field (canonical English)
    openalex_fields: List[int]
    openalex_subfields: List[int]

    @property
    def column(self) -> str:
        """Column with the number of works of an author in this field, e.g. works_computer_science."""
        return f"works_{field_slug(self.field)}"

    def sql_condition(self, field_column: str = 'field_id', subfield_column: str = 'subfield_id') -> str:
        parts = []
        if self.openalex_fields:
            parts.append(f"{field_column} IN ({', '.join(map(str, self.openalex_fields))})")
        if self.openalex_subfields:
            parts.append(f"{subfield_column} IN ({', '.join(map(str, self.openalex_subfields))})")
        if not parts:
            raise ValueError(f"Field '{self.field}' maps to no OpenAlex field or subfield")
        return '(' + ' OR '.join(parts) + ')'


def field_slug(field: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', field.lower()).strip('_')


def load_field_mapping(file_path: Path | str = cons.DEFAULT_FIELD_MAPPING_FILE) -> List[FieldMapping]:
    """Mappings in the canonical prompt-field order; fails if a prompt field is missing."""
    raw: Dict = ios.load_json(file_path)
    mappings = {name: FieldMapping(name, [f['id'] for f in spec.get('openalex_fields', [])],
                                   [s['id'] for s in spec.get('openalex_subfields', [])])
                for name, spec in raw.items() if not name.startswith('_')}
    prompt_fields = cons.VARIABLE_VALUES[cons.VAR_FIELD]
    missing = [f for f in prompt_fields if f not in mappings]
    if missing:
        raise ValueError(f"Prompt fields without a mapping in {file_path}: {missing}")
    return [mappings[f] for f in prompt_fields]
