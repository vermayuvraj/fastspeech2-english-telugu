import json

from fs2.config import ROOT
from fs2.frontend import SYMBOLS, expand_numbers, normalize, te_number


def test_all_assignment_sentences_are_preserved_and_encodable():
    ids = []
    for language in ["English", "Telugu"]:
        rows = json.loads((ROOT / "test_sentences" / f"{language}.json").read_text(encoding="utf-8"))
        assert len(rows) == 5
        for row in rows:
            front = normalize(row["text"], language)
            assert front["raw_text"] == row["text"]
            assert not any(c.isdigit() for c in front["normalized_text"])
            assert all(0 < token < len(SYMBOLS) for token in front["token_ids"])
            ids.append(row["id"])
    assert len(set(ids)) == 10


def test_malformed_words_are_not_silently_repaired():
    assert normalize("The rhyth  m", "English")["words"] == ["the", "rhyth", "m"]
    assert normalize("అడ విలో", "Telugu")["words"] != normalize("అడవిలో", "Telugu")["words"]


def test_clock_date_and_telugu_numbers_are_expanded():
    text = expand_numbers("10:45 AM on 12 November 2026", "English")
    assert "ten forty-five a m" in text
    assert "two thousand" in text
    text = expand_numbers("2026 నవంబర్ 12వ 10:45", "Telugu")
    assert "రెండు వేల ఇరవై ఆరు" in text
    assert "పన్నెండవ" in text
    assert "పది గంటల నలభై ఐదు నిమిషాలు" in text
    assert te_number(45) == "నలభై ఐదు"


def test_ordinary_am_is_not_an_am_pm_abbreviation():
    assert expand_numbers("I am here.", "English") == "I am here."
