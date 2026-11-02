import pytest

from fs2.preprocess import speaker_for_row


def test_profile_conditioning_is_deterministic_and_uses_all_fields():
    config = {"dataset": {"speaker_profile_fields": ["gender", "region"]}}
    first = speaker_for_row({"gender": "Female", "region": "A"}, config)
    assert first == speaker_for_row({"region": "A", "gender": "Female"}, config)
    assert first != speaker_for_row({"gender": "Female", "region": "B"}, config)
    with pytest.raises(ValueError, match="column"):
        speaker_for_row({"gender": "Female"}, config)


def test_gender_is_not_silently_used_as_a_speaker_identifier():
    with pytest.raises(ValueError, match="speaker"):
        speaker_for_row({"gender": "Female"}, {"dataset": {}})
    assert speaker_for_row({"gender": 1}, {"dataset": {"speaker_field": "gender"}}) == "1"
