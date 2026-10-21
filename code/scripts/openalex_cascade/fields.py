"""
fields.py — Which of our 6 fields (and 12 subfields) an OpenAlex author belongs to.

Two published classifications of OpenAlex are used, none made up here:

  topics    OpenAlex's current hierarchy (domain → field → subfield → topic),
            built on Scopus' ASJC categories. FIELD_MAP / SUBFIELD_TO_FIELD
            name the OpenAlex fields that are one of ours. "Biology" is the
            Life Sciences domain except Pharmacology; Sociology is the
            "Sociology and Political Science" subfield of Social Sciences.
  concepts  The level-0 concepts OpenAlex inherited from Microsoft Academic
            Graph, still in the snapshot's author records (`x_concepts`).
            Their names are ours (CONCEPT_MAP).

Validated on 5,671 authors with a name that is unique in both Semantic Scholar
and OpenAlex and the same first-publication year (±2): the main topics field
agrees with Semantic Scholar's field for 72.4% of them, the main concept for
77.8%, both together (when they agree) for 83.9%. See
notebooks/analysis/openalex_field_validation.ipynb.

SUBFIELDS gives, for each subfield of our prompts, the OpenAlex subfields that
correspond to it; it is only used to break ties. Forensic Psychology,
Criminology and Family have no OpenAlex counterpart.
"""

# OpenAlex field → our field (topics hierarchy).
FIELD_MAP = {
    "Physics and Astronomy": "Physics",
    "Computer Science": "Computer Science",
    "Mathematics": "Mathematics",
    "Psychology": "Psychology",
    "Biochemistry, Genetics and Molecular Biology": "Biology",
    "Agricultural and Biological Sciences": "Biology",
    "Immunology and Microbiology": "Biology",
    "Neuroscience": "Biology",
}

# OpenAlex subfield → our field, for OpenAlex fields that are only partly ours.
SUBFIELD_TO_FIELD = {
    "Sociology and Political Science": "Sociology",
}

# Level-0 concept (inherited from Microsoft Academic Graph) → our field.
CONCEPT_MAP = {
    "Biology": "Biology",
    "Computer science": "Computer Science",
    "Mathematics": "Mathematics",
    "Physics": "Physics",
    "Psychology": "Psychology",
    "Sociology": "Sociology",
}

# Our subfield → the OpenAlex subfields that correspond to it.
SUBFIELDS = {
    "Anatomy": {"Anatomy"},
    "Neuroscience": {"Behavioral Neuroscience", "Biological Psychiatry",
                     "Cellular and Molecular Neuroscience", "Cognitive Neuroscience",
                     "Developmental Neuroscience", "Endocrine and Autonomic Systems",
                     "Neurology", "Sensory Systems"},
    "Artificial Intelligence": {"Artificial Intelligence"},
    "Software Engineering": {"Software"},
    "Number theory": {"Algebra and Number Theory"},
    "Topology": {"Geometry and Topology"},
    "Condensed Matter": {"Condensed Matter Physics"},
    "Education": {"Education"},
    "Social Psychology": {"Social Psychology"},
}


def study_field(field: str | None, subfield: str | None) -> str | None:
    """Our field for an OpenAlex (field, subfield), or None if it is none of the 6."""
    return SUBFIELD_TO_FIELD.get(subfield) or FIELD_MAP.get(field)
