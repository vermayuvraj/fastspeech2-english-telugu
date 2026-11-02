from types import SimpleNamespace

import numpy as np
import pytest

from fs2.alignment import spans_to_durations
from fs2.split import assert_disjoint, select_splits


def test_duplicate_audio_and_shared_prompts_never_cross_splits():
    rows = [{"id": str(i), "audio_sha256": f"hash-{i}", "normalized_text": f"prompt-{i // 2}", "speaker": str(i % 2), "duration_seconds": 12.0} for i in range(100)]
    rows.append({**rows[0], "id": "duplicate"})
    first = select_splits(rows, 0.1, 0.03, 17)
    second = select_splits(list(reversed(rows)), 0.1, 0.03, 17)
    assert first == second
    assert first[2]["duplicate_audio_removed"] == 1
    assert_disjoint(*first[:2])
    assert first[2]["duration_seconds"]["train"] >= 360
    assert first[2]["duration_seconds"]["validation"] >= 108


def test_overlap_detection_checks_audio_not_only_ids():
    row = {"id": "1", "audio_sha256": "same", "normalized_text": "one"}
    with pytest.raises(ValueError, match="audio_sha256"):
        assert_disjoint([row], [{**row, "id": "2", "normalized_text": "two"}])


def test_durations_partition_frames_and_keep_real_interword_gap():
    def span(start, end):
        return SimpleNamespace(start=start, end=end)
    words = ["ab", "c"]
    spans = [[span(2, 4), span(5, 7)], [span(10, 12)]]
    tokens, duration = spans_to_durations(words, spans, 15, 1.5, 129)
    assert tokens == ["<sil>", "a", "b", "|", "c", "<sil>"]
    assert len(duration) == len(tokens)
    assert duration.sum() == 129
    assert np.all(duration >= 0)
    assert duration[3] > 0
