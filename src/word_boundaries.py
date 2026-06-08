"""Refine word spans so punctuation glued to letters becomes a word boundary.

Binder's mask gate (run_ner.py, src/inference.py) only allows predictions whose
start/end character offsets appear in the example's word_start_chars /
word_end_chars. NLTK's word tokenizer keeps several punctuation classes glued
to adjacent letters/digits (em-dash between digits in date ranges, hyphens in
compounds, leading apostrophes, periods between letters), so gold entity
boundaries inside those joined tokens fail the gate and can never be emitted.

This module takes whatever spans the base tokenizer produced and adds more
boundaries by splitting each span on the relevant punctuation. Extra word
boundaries are harmless to the mask gate; missing ones silently kill recall.
"""

import re

_ALWAYS_SPLIT = "—–'\"«»„“”‘’`"
_LETTER = r"A-Za-zЀ-ӿ"

_SPLIT_RE = re.compile(
    r"[" + re.escape(_ALWAYS_SPLIT) + r"]"
    r"|(?<=\w)-(?=\w)"
    r"|(?<=[" + _LETTER + r"])\.(?=[" + _LETTER + r"]|\d|$)"
    r"|(?<=\d)\.(?=[" + _LETTER + r"]|$)"
    r"|(?<=\w)[:|](?=\w)"
    r"|(?<=[" + _LETTER + r"])(?=\d)"
    r"|(?<=\d)(?=[" + _LETTER + r"])",
    re.UNICODE,
)


def refine_word_spans(text, spans):
    """Split each (start, end) span on punctuation that should be a word break.

    Punctuation characters that trigger a split are emitted as their own
    one-character spans, so their character positions become valid word
    boundaries for mask construction.
    """
    refined = []
    for s, e in spans:
        chunk = text[s:e]
        if not _SPLIT_RE.search(chunk):
            refined.append((s, e))
            continue
        cursor = 0
        for m in _SPLIT_RE.finditer(chunk):
            ms, me = m.span()
            if ms > cursor:
                refined.append((s + cursor, s + ms))
            if me > ms:
                refined.append((s + ms, s + me))
            cursor = me
        if cursor < len(chunk):
            refined.append((s + cursor, s + len(chunk)))
    return refined
