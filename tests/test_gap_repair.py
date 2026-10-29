import numpy as np

from src.preprocessing.gap_repair import repair_hand_gaps


def test_repairs_only_short_bracketed_gaps_with_reliable_wrist():
    count = 11
    hand = np.zeros((count, 21, 3), dtype=np.float32)
    pose = np.zeros((count, 33, 3), dtype=np.float32)
    pose[:, 15, 0] = 0.2
    pose[:, 16, 0] = 0.8
    pose[:, 15, 1] = np.arange(count) / 10
    pose[:, 16, 1] = np.arange(count) / 10
    observed = np.array([False, True, False, False, True, False, False,
                         False, True, False, True])
    for frame in np.flatnonzero(observed):
        hand[frame, :, 0] = 0.2
        hand[frame, :, 1] = frame / 10
    repaired, imputed = repair_hand_gaps(
        hand, observed, pose, np.ones(count, dtype=bool),
        wrist_index=15, other_wrist_index=16, max_gap=2,
    )
    assert np.array_equal(np.flatnonzero(imputed), [2, 3, 9])
    assert np.isclose(repaired[2, 0, 1], 0.2)
    assert np.isclose(repaired[3, 0, 1], 0.3)
    assert np.array_equal(repaired[[0, 5, 6, 7]], hand[[0, 5, 6, 7]])
    assert np.array_equal(repaired[observed], hand[observed])


def test_rejects_wrong_hand_or_missing_pose():
    hand = np.zeros((3, 21, 3), dtype=np.float32)
    hand[[0, 2], :, 0] = 0.8
    pose = np.zeros((3, 33, 3), dtype=np.float32)
    pose[:, 15, 0] = 0.2
    pose[:, 16, 0] = 0.8
    observed = np.array([True, False, True])
    repaired, imputed = repair_hand_gaps(
        hand, observed, pose, np.ones(3, dtype=bool),
        wrist_index=15, other_wrist_index=16,
    )
    assert not imputed.any()
    assert np.array_equal(repaired, hand)
    hand[[0, 2], :, 0] = 0.2
    _, imputed = repair_hand_gaps(
        hand, observed, pose, np.array([True, False, True]),
        wrist_index=15, other_wrist_index=16,
    )
    assert not imputed.any()


def test_rejects_out_of_frame_endpoint_and_non_linear_wrist_motion():
    hand = np.zeros((3, 21, 3), dtype=np.float32)
    hand[[0, 2], :, :2] = [0.2, 0.5]
    pose = np.zeros((3, 33, 3), dtype=np.float32)
    pose[:, 15, :2] = [0.2, 0.5]
    pose[:, 16, :2] = [0.8, 0.5]
    observed = np.array([True, False, True])
    kwargs = dict(wrist_index=15, other_wrist_index=16)
    hand[0, 20, 0] = -0.1
    repaired, imputed = repair_hand_gaps(hand, observed, pose, np.ones(3, dtype=bool), **kwargs)
    assert not imputed.any()
    assert np.array_equal(repaired, hand)  # never clamp measured landmarks
    hand[0, 20, 0] = 0.2
    pose[1, 15, 1] = 0.9  # endpoints fit but the real wrist moves away in between
    _, imputed = repair_hand_gaps(hand, observed, pose, np.ones(3, dtype=bool), **kwargs)
    assert not imputed.any()
