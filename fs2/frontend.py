"""Auditable normalization shared by training, alignment and inference.

The input sentence is never overwritten. Romanized graphemes are linguistic
units, not IPA phonemes. Extra spaces and malformed words are not spell-corrected.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from num2words import num2words

SYMBOLS = ["<pad>", "<sil>", "|"] + list("abcdefghijklmnopqrstuvwxyz'")
SYMBOL_TO_ID = {symbol: i for i, symbol in enumerate(SYMBOLS)}

# Explicit Telugu number rules, independent of optional num2words language support.
TE_SMALL = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది", "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పంతొమ్మిది"]
TE_TENS = {20: "ఇరవై", 30: "ముప్పై", 40: "నలభై", 50: "యాభై", 60: "అరవై", 70: "డెబ్బై", 80: "ఎనభై", 90: "తొంభై"}


def te_number(n: int) -> str:
    if n < 0:
        return "మైనస్ " + te_number(-n)
    if n < 20:
        return TE_SMALL[n]
    if n < 100:
        q, r = divmod(n, 10)
        return TE_TENS[q * 10] + (" " + TE_SMALL[r] if r else "")
    if n < 1000:
        q, r = divmod(n, 100)
        head = "వంద" if q == 1 else te_number(q) + " వందల"
        return head + (" " + te_number(r) if r else "")
    for scale, singular, plural in [(10000000, "కోటి", "కోట్ల"), (100000, "లక్ష", "లక్షల"), (1000, "వెయ్యి", "వేల")]:
        if n >= scale:
            q, r = divmod(n, scale)
            head = singular if q == 1 else te_number(q) + " " + plural
            return head + (" " + te_number(r) if r else "")
    raise ValueError(n)


def number_words(n: int, language: str) -> str:
    return te_number(n) if language == "Telugu" else num2words(n, lang="en")


def te_ordinal(n: int) -> str:
    if n == 1:
        return "మొదటి"
    tens = {20: "ఇరవయ్యవ", 30: "ముప్పయ్యవ", 40: "నలభయ్యవ", 50: "యాభయ్యవ", 60: "అరవయ్యవ", 70: "డెబ్బయ్యవ", 80: "ఎనభయ్యవ", 90: "తొంభయ్యవ"}
    if n in tens:
        return tens[n]
    cardinal = te_number(n)
    return cardinal.rstrip("ుి") + "వ"


def expand_numbers(text: str, language: str) -> str:
    text = "".join(str(unicodedata.digit(c)) if c.isdecimal() else c for c in text)

    def clock(match):
        hour, minute = int(match[1]), int(match[2])
        if hour > 23 or minute > 59:
            raise ValueError(f"Invalid clock time: {match[0]}")
        # Read the written time literally; the original AM/PM is expanded below.
        if language == "Telugu":
            suffix = "నిమిషాలకు" if match[3] else "నిమిషాలు"
            return te_number(hour) + " గంటల " + te_number(minute) + " " + suffix if minute else te_number(hour) + (" గంటలకు" if match[3] else " గంటలు")
        middle = " oh " if 0 < minute < 10 else " "
        if not minute:
            return number_words(hour, language) + (" గంటలు" if language == "Telugu" else " o'clock")
        return number_words(hour, language) + middle + number_words(minute, language)

    time_pattern = r"(?<!\d)(\d{1,2}):(\d{2})(\s+గంటలకు)?" if language == "Telugu" else r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)"
    text = re.sub(time_pattern, clock, text)
    if language == "English":
        text = re.sub(r"\bAM\b", "a m", text)
        text = re.sub(r"\bPM\b", "p m", text)
        text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: num2words(int(m[1]), to="ordinal"), text, flags=re.I)
        months = "January|February|March|April|May|June|July|August|September|October|November|December"
        text = re.sub(r"\b(\d{1,2})\s+(" + months + r")\b", lambda m: num2words(int(m[1]), to="ordinal") + " of " + m[2], text, flags=re.I)
    else:
        # In 12వ, preserve the ordinal marker instead of dropping the number.
        text = re.sub(r"(\d+)వ", lambda m: te_ordinal(int(m[1])), text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\d+\.\d+", lambda m: number_words(int(m[0].split('.')[0]), language) + (" దశాంశం " if language == "Telugu" else " point ") + " ".join(number_words(int(d), language) for d in m[0].split('.')[1]), text)
    return re.sub(r"\d+", lambda m: number_words(int(m[0]), language), text)


@lru_cache(maxsize=1)
def romanizer():
    from uroman import Uroman
    return Uroman()


def normalize(text: str, language: str) -> dict:
    if language not in {"English", "Telugu"}:
        raise ValueError(f"Unsupported language: {language}")
    expanded = expand_numbers(unicodedata.normalize("NFC", text).replace("’", "'"), language)
    roman = romanizer().romanize_string(expanded, lcode="tel" if language == "Telugu" else "eng")
    roman = re.sub(r"[^a-z' ]", " ", roman.lower())
    words = re.sub(r"\s+", " ", roman).strip().split()
    if not words or not any(re.search("[a-z]", word) for word in words):
        raise ValueError("Transcript has no alignable letters")
    tokens = ["<sil>"]
    for i, word in enumerate(words):
        if i:
            tokens.append("|")
        tokens.extend(word)
    tokens.append("<sil>")
    return {"raw_text": text, "expanded_text": expanded, "normalized_text": " ".join(words), "words": words, "tokens": tokens, "token_ids": [SYMBOL_TO_ID[t] for t in tokens]}
