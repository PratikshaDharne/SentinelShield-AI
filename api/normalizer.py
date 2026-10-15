"""Input Gateway: normalization + deobfuscation.

IMPORTANT: the normalized text is used for DETECTION only. Decoded payloads
(base64/hex/rot13) are inspected as DATA and are never executed or obeyed.
"""
from __future__ import annotations
import base64, binascii, codecs, re, unicodedata
from dataclasses import dataclass, field

# Cyrillic / Greek / fullwidth look-alikes -> Latin
_HOMOGLYPHS = {
    "а": "a", "в": "b", "с": "c", "е": "e", "н": "h", "і": "i", "ј": "j", "к": "k",
    "м": "m", "о": "o", "р": "p", "ѕ": "s", "т": "t", "у": "y", "х": "x", "ԁ": "d",
    "ԛ": "q", "ԝ": "w", "ɡ": "g", "ӏ": "l", "ο": "o", "ν": "v", "ρ": "p", "α": "a",
    "ε": "e", "ι": "i", "κ": "k", "τ": "t", "υ": "u", "χ": "x", "ɑ": "a",
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "Ј": "J", "К": "K",
    "М": "M", "О": "O", "Р": "P", "Ѕ": "S", "Т": "T", "У": "Y", "Х": "X", "Ү": "Y",
    "Ο": "O", "Α": "A", "Β": "B", "Ε": "E", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X", "Ζ": "Z",
}
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­‪‫‬‭‮"), None)
_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i", "8": "b"}
_B64_RE = re.compile(r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{24,}={0,2})(?![A-Za-z0-9+/=])")
_HEX_RE = re.compile(r"(?<![0-9A-Fa-f])((?:[0-9A-Fa-f]{2}[\s:]?){12,})(?![0-9A-Fa-f])")
_WS = re.compile(r"[ \t\r\f\v]+")
_SPACED = re.compile(r"(?<![A-Za-z])(?:[A-Za-z] ){2,}[A-Za-z](?![A-Za-z])")          # i g n o r e
_DOTTED = re.compile(r"(?<![A-Za-z])(?:[A-Za-z][.\-_*]){3,}[A-Za-z](?![A-Za-z])")     # i.g.n.o.r.e
_COMMON = set("the and you your all previous instructions ignore reveal system prompt explain what how is of to a in "
              "disable safety hidden then rules password configuration decode execute".split())


@dataclass
class NormResult:
    raw: str
    normalized: str                      # for detection
    decoded_views: list[str] = field(default_factory=list)   # decoded payloads (data only)
    payload_spans: list[str] = field(default_factory=list)   # raw encoded strings found
    flags: list[str] = field(default_factory=list)           # obfuscation flags


def _printable_ratio(s: str) -> float:
    return sum(ch.isprintable() or ch in "\n\t" for ch in s) / max(1, len(s))


def _looks_like_text(s: str) -> bool:
    if len(s) < 8 or _printable_ratio(s) < 0.95:
        return False
    words = re.findall(r"[a-z]+", s.lower())
    return bool(words) and sum(w in _COMMON for w in words) / len(words) >= 0.15 or len(words) >= 4 and \
        sum(c.isalpha() or c.isspace() for c in s) / len(s) > 0.9


_ORDINAL = re.compile(r"^\d+(st|nd|rd|th|k|g|x|d|gb|mb|kb)$", re.I)


def _deleet_token(tok: str, aggressive: bool) -> str:
    """De-leet one token. Conservative mode protects identifiers like gpt4/mp3/h2o."""
    core = tok.strip(".,;:?\"'()[]{}")
    letters = sum(c.isalpha() for c in core)
    leets = sum(c in _LEET for c in core)
    if leets == 0 or core.isdigit() or _ORDINAL.match(core) or len(core) < 2:
        return tok
    if re.search(r"@[\w-]+\.\w|://|^[/~.]|\\|\.(?:com|org|net|io|py|js|env|txt|json)\b", core, re.I):
        return tok      # emails, URLs, paths, filenames are not leetspeak
    if letters < (1 if aggressive else 2):
        return tok
    if not aggressive and re.fullmatch(r"[A-Za-z]+\d+", core):
        return tok  # gpt4, mp3, h2o
    out = "".join(_LEET.get(c, c) for c in core)
    return tok.replace(core, out)


def _common_hits(s: str) -> int:
    return sum(w in _COMMON for w in re.findall(r"[a-z]+", s.lower()))


def normalize(text: str) -> NormResult:
    raw = text or ""
    flags: list[str] = []
    t = unicodedata.normalize("NFKC", raw)
    if t.translate(_ZERO_WIDTH) != t:
        flags.append("ZERO_WIDTH")
        t = t.translate(_ZERO_WIDTH)
    mapped = "".join(_HOMOGLYPHS.get(c, c) for c in t)
    if mapped != t:
        flags.append("UNICODE_HOMOGLYPH")
        t = mapped
    # letter-spacing obfuscation: "i g n o r e", "i.g.n.o.r.e"
    collapsed = _SPACED.sub(lambda m: m.group(0).replace(" ", ""), t)
    collapsed = _DOTTED.sub(lambda m: re.sub(r"[.\-_*]", "", m.group(0)), collapsed)
    if collapsed != t:
        flags.append("LETTER_SPACING"); t = collapsed
    # 1) find encoded payloads FIRST; keep them as data, replace with a placeholder
    decoded, spans = [], []
    for m in list(_B64_RE.finditer(t)):
        s = m.group(1)
        try:
            d = base64.b64decode(s + "=" * (-len(s) % 4), validate=True).decode("utf-8", "strict")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if _looks_like_text(d):
            decoded.append(d); spans.append(s); flags.append("BASE64")
            t = t.replace(s, " [ENCODED_PAYLOAD] ")
    for m in list(_HEX_RE.finditer(t)):
        s = m.group(1)
        try:
            d = bytes.fromhex(re.sub(r"[\s:]", "", s)).decode("utf-8", "strict")
        except ValueError:
            continue
        if _looks_like_text(d):
            decoded.append(d); spans.append(s.strip()); flags.append("HEX")
            t = t.replace(s, " [ENCODED_PAYLOAD] ")
    # 2) leetspeak (aggressive only when the text is clearly leet-written)
    toks = re.split(r"(\s+)", t)
    strong = sum(1 for x in toks if not x.isspace() and _deleet_token(x, False) != x)
    aggressive = strong >= 2
    deleeted = "".join(_deleet_token(x, aggressive) if not x.isspace() else x for x in toks)
    if deleeted != t:
        flags.append("LEETSPEAK")
        t = deleeted
    # 3) rot13 view (data only), only if it clearly reveals english where the text had none
    rot = codecs.decode(t, "rot_13")
    if _common_hits(rot) >= 3 and _common_hits(t) <= 1:
        decoded.append(rot); flags.append("ROT13")
    t = _WS.sub(" ", t).strip()
    return NormResult(raw=raw, normalized=t, decoded_views=decoded, payload_spans=spans, flags=sorted(set(flags)))
