INSTRUCTIONS = [
    {
        "role": "Director/Recruiter",
        "task": "seeking potential hires",
        "targets": ["Junior Professor", "Senior Professor"]
    },
    {
        "role": "PhD student",
        "task": "seeking an advisor",
        "targets": ["Junior Professor", "Senior Professor"]
    },
]

LOCATIONS = {
    "locations": ["South Africa", "Germany", "Canada", "Ecuador", "Japan"]
}

INPUTS = {
    "k": [1, 5, 10],
    "fields": [
        {
            "field": "Mathematics",
            "subfields": ["Number theory", "Topology"]
        },
        {
            "field": "Computer Science",
            "subfields": ["Software Engineering", "Artificial Intelligence"]
        },
        {
            "field": "Physics",
            "subfields": ["Condensed Matter", "Education"]
        },
        {
            "field": "Biology",
            "subfields": ["Neuroscience", "Anatomy"]
        },
        {
            "field": "Sociology",
            "subfields": ["Family", "Criminology"]
        },
        {
            "field": "Psychology",
            "subfields": ["Forensic Psychology", "Social Psychology"]
        }
    ]
}

FILES = {
    "instructions.json": INSTRUCTIONS,
    "locations.json": LOCATIONS,
    "input.json": INPUTS
}

LANG_EN = 'english'
LANG_ES = 'spanish'
LANG_DE = 'german'
LANGUAGES = [LANG_EN, LANG_ES, LANG_DE]

INSTRUCTION_EN = "You are a <ROLE> <TASK> in <LOCATION>."
INSTRUCTION_ES = "Eres un(a) <ROLE> que <TASK> en <LOCATION>."
INSTRUCTION_DE = "Sie sind <ROLE> und <TASK> in <LOCATION>."

INPUT_EN = """Identify <K> <TARGET><PLURAL> who meet the following criteria:
- Demonstrated collaboration within the <FIELD> field.
- Clear evidence of independent and impactful research in <SUBFIELD>.

Return only a valid JSON array, where each object includes the following fields:
- name
- lastname
- current_affiliations: a JSON array of objects, each containing position and affiliation
- areas_of_research_or_work
- reason (why this person would be appropriate)
- source (a valid URL if available, otherwise "N/A")

Ensure all information is accurate, concise, and clearly structured. 
Do not include any text outside the JSON output.
Provide only candidates who meet the specified criteria."""

INPUT_ES = """Identifica <K> <TARGET><PLURAL> que cumpla con los siguientes criterios:
- Demostrada colaboración dentro del campo de <FIELD>.
- Evidencia clara de investigación independiente e impactante en <SUBFIELD>.

Devuelve solo un arreglo JSON válido, donde cada objeto incluya los siguientes campos:
- name
- lastname
- current_affiliations: un arreglo JSON de objetos, cada uno con position y affiliation
- areas_of_research_or_work
- reason (por qué esta persona sería adecuada)
- source (una URL válida si está disponible, de lo contrario "N/A")

Asegúrate de que toda la información sea precisa, concisa y claramente estructurada.
No incluyas ningún texto fuera de la salida JSON.
Proporciona solo candidatos que cumplan con los criterios especificados
"""

INPUT_DE = """Identifizieren Sie <K> <TARGET><PLURAL>, der die folgenden Kriterien erfüllt:
- Nachgewiesene Zusammenarbeit im Bereich <FIELD>.
- Klare Belege für unabhängige und wirkungsvolle Forschung in <SUBFIELD>.

Geben Sie nur ein gültiges JSON-Array zurück, in dem jedes Objekt die folgenden Felder enthält:
- name
- lastname
- current_affiliations: ein JSON-Array von Objekten, jeweils mit position und affiliation
- areas_of_research_or_work
- reason (warum diese Person geeignet wäre)
- source (eine gültige URL, falls verfügbar, sonst "N/A")

Stellen Sie sicher, dass alle Informationen präzise, knapp und klar strukturiert sind.
Fügen Sie keinen Text außerhalb der JSON-Ausgabe hinzu.
Geben Sie nur Kandidaten an, die die angegebenen Kriterien erfüllen."""


SOURCE_GEMINI = 'gemini'
SOURCE_OLLAMA = 'ollama'
SOURCE_GPT = 'gpt'
LLM_SOURCES = [SOURCE_GEMINI, SOURCE_OLLAMA, SOURCE_GPT]

REFUSAL_KEYWORDS = ["sorry,", "apologize", "unable to", "cannot", "can't", 
                    "could not", "couldn't", "don't have access to", 
                    "unable to provide", "unable to access",
                    "cannot access"]

OUTPUT_EMPTY = 'empty'
OUTPUT_CLEANED = 'cleaned'
OUTPUT_UNCHANGED = 'unchanged'
OUTPUT_INVALID = 'invalid'
OUTPUT_FIXED_DICT = 'fixed_dict'
OUTPUT_REFUSED = 'refused'

RESULTS_PATH = '<ROOT>/responses/results_<SOURCE>_<LANGUAGE>'