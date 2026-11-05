"""
Multi-label tags: encoding as one joined string column plus one boolean column per tag.

    tag_columns(['wrapped', 'correct'], cons.RESPONSE_TAGS, 'response_')
    -> {'response_tags': 'correct|wrapped', 'response_correct': True, 'response_wrapped': True, ...False}
"""

from typing import Dict, Iterable, List

from utils import constants as cons


def ordered(tags: Iterable[str], vocabulary: List[str]) -> List[str]:
    """Unique ``tags`` in vocabulary order (unknown tags raise, to catch typos early)."""
    tags = set(tags)
    unknown = tags.difference(vocabulary)
    if unknown:
        raise ValueError(f"Unknown tags: {sorted(unknown)}")
    return [tag for tag in vocabulary if tag in tags]


def join(tags: Iterable[str], vocabulary: List[str]) -> str:
    """Tags joined by the separator, in vocabulary order."""
    return cons.TAG_SEPARATOR.join(ordered(tags, vocabulary))


def split(joined: str) -> List[str]:
    """Inverse of :func:`join`."""
    return joined.split(cons.TAG_SEPARATOR) if joined else []


def tag_columns(tags: Iterable[str], vocabulary: List[str], prefix: str) -> Dict[str, object]:
    """The joined-tags column and one boolean column per vocabulary tag."""
    present = set(ordered(tags, vocabulary))
    return {f"{prefix}tags": join(present, vocabulary),
            **{f"{prefix}{tag}": tag in present for tag in vocabulary}}


def column_names(vocabulary: List[str], prefix: str) -> List[str]:
    """Names of the columns produced by :func:`tag_columns`, in order."""
    return [f"{prefix}tags"] + [f"{prefix}{tag}" for tag in vocabulary]
