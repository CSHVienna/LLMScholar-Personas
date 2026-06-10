import ast
import re
import unicodedata

try:
    from prompt import constants as cons
except ImportError:  # importing as libs.utils.text (PYTHONPATH=code/)
    from libs.prompt import constants as cons


def clean_content(text):
    """Cleans the output text by fixing umlauts, removing diacritics/tildes, and removing unnecessary quotation marks."""

    original = text

    # Extract content from markdown code fences (```json ... ```) even with surrounding text
    m = re.search(r"```[a-zA-Z]*\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()

    # umlauts
    for v in "aeiou":
        text = text.replace(f'\\"{v}', f"{v}̈")
        text = text.replace(f'\\"{v.upper()}', f"{v.upper()}̈")

    # text in quotation marks
    text = re.sub(r'\\\"([^"]+)\\\"', r"\1", text)

    # illegal surrogate — raw_unicode_escape avoids the Python 3 bug where
    # codecs.decode(str) uses UTF-8 internally, corrupting é→Ã©, ñ→Ã±, etc.
    decoded = text.encode("raw_unicode_escape").decode("unicode_escape")
    text = re.sub(r"[\ud800-\udfff]", "", decoded)

    # quotation marks
    text = text.replace('\\""', '"')
    text = text.replace('\\",', '",')
    text = text.replace('\\"', '"')

    text = text.replace('},\n    "', '",\n    "')

    # Remove diacritics and tildes (e.g. á→a, ü→u, ñ→n) for ground truth matching
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    return text, cons.OUTPUT_CLEANED if text != original else cons.OUTPUT_UNCHANGED


def parse_valid_dicts(text):
    """Parses valid dictionaries from a text representation of a list of dictionaries."""

    def split_top_level(s):
        items = []
        depth = 0
        start = 0

        for i, ch in enumerate(s):
            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
            elif ch == "," and depth == 0:
                items.append(s[start:i].strip())
                start = i + 1

        items.append(s[start:].strip())
        return items

    m = re.search(r"\[(.*)", text, re.DOTALL)
    if not m:
        raise ValueError("No list found")

    list_text = m.group(1)
    valid_items = []

    for item in split_top_level(list_text):
        try:
            obj = ast.literal_eval(item)
        except Exception:
            continue

        if isinstance(obj, dict):
            valid_items.append(obj)

    return valid_items
