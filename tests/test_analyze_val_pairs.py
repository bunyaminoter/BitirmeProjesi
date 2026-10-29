"""Paired val comparison must reject mislabeled or incomplete prediction files."""

import json

import pytest

from scripts.analyze_val_pairs import analyze, exact_mcnemar_p


def test_pair_analysis_checks_official_ids_and_labels(tmp_path):
    labels = tmp_path / "val_labels.csv"
    labels.write_text("signer1_sample1,8\nsigner2_sample2,14\n", encoding="utf-8")
    first = tmp_path / "base.json"
    second = tmp_path / "candidate.json"
    metadata = {"split": "val", "class_ids": [8, 14]}
    base_records = [
        {"video_id": "signer1_sample1_color", "true_label": 0, "pred_label": 0, "top1_confidence": 0.9},
        {"video_id": "signer2_sample2_color", "true_label": 1, "pred_label": 0, "top1_confidence": 0.7},
    ]
    candidate_records = [
        {"video_id": "signer1_sample1_color", "true_label": 0, "pred_label": 1, "top1_confidence": 0.6},
        {"video_id": "signer2_sample2_color", "true_label": 1, "pred_label": 1, "top1_confidence": 0.8},
    ]
    first.write_text(json.dumps({**metadata, "predictions": base_records}), encoding="utf-8")
    second.write_text(json.dumps({**metadata, "predictions": candidate_records}), encoding="utf-8")

    result = analyze(first, second, labels)
    assert result["paired_counts"] == {
        "both_correct": 0, "base_only": 1, "candidate_only": 1, "both_wrong": 0
    }
    assert result["net_correct_videos"] == 0
    assert exact_mcnemar_p(0, 6) == pytest.approx(0.03125)

    candidate_records[0]["true_label"] = 1
    second.write_text(json.dumps({**metadata, "predictions": candidate_records}), encoding="utf-8")
    with pytest.raises(ValueError, match="Label mismatch"):
        analyze(first, second, labels)
