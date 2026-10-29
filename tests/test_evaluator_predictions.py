"""Per-video prediction export must keep sample IDs and labels aligned."""

import torch

from src.evaluation.evaluator import Evaluator


class _LogitModel(torch.nn.Module):
    def forward(self, batch):
        return batch["logits"]


def test_evaluator_prediction_records_follow_batch_order():
    batch = {
        "logits": torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
        "labels": torch.tensor([0, 1]),
        "video_ids": ["video_a", "video_b"],
    }
    evaluator = Evaluator(_LogitModel(), num_classes=2, collect_predictions=True)

    result = evaluator.evaluate([batch])

    assert result.accuracy == 1.0
    assert [record["video_id"] for record in evaluator.prediction_records] == [
        "video_a", "video_b"
    ]
    assert [record["true_label"] for record in evaluator.prediction_records] == [0, 1]
    assert [record["pred_label"] for record in evaluator.prediction_records] == [0, 1]
    assert all(0.5 < record["top1_confidence"] < 1.0 for record in evaluator.prediction_records)
    for record in evaluator.prediction_records:
        scores = record["class_probabilities"]
        assert len(scores) == 2
        assert abs(sum(scores) - 1.0) < 1e-6
        assert scores.index(max(scores)) == record["pred_label"]
        assert abs(max(scores) - record["top1_confidence"]) < 1e-6
