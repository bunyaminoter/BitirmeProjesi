import numpy as np
import pytest

from src.preprocessing.wrist_fallback_crop import DEFAULT_CENTER_REPAIR, fallback_geometry


def scene(observed):
    count = len(observed)
    hands = np.zeros((count, 21, 3), np.float32)
    hands[:, :, 0] = np.linspace(.35, .45, 21)
    hands[:, :, 1] = np.linspace(.4, .6, 21)
    hands[:, 0, :2] = [.4, .5]
    pose = np.full((count, 33, 3), .5, np.float32)
    pose[:, [15, 17, 19, 21], 0] = .4
    pose[:, [16, 18, 20, 22], 0] = .9
    mask = np.array(observed, bool)
    hands[~mask] = 0
    return dict(hands=hands, observed=mask, pose=pose, pose_ok=np.ones(count, bool),
                side='left', width=200, height=100, box_scale=1.5,
                center_repair=dict(DEFAULT_CENTER_REPAIR))


def test_repairs_bad_center_and_pose_missing_only_in_short_interior_gap():
    data = scene([False, True, False, False, True, False, True])
    data['pose'][2:4, [15,17,19,21], :2] = [.1, .9]
    data['pose_ok'][2] = False
    before_hands, before_pose = data['hands'].copy(), data['pose'].copy()
    source, centers, boxes = fallback_geometry(**data)
    assert source.tolist()==[1,0,3,3,0,1,0]  # leading gap and stable pose untouched
    assert np.allclose(centers[2:4], [[80.5,50.5],[80.5,50.5]])
    assert (boxes[2:4, 2] > 0).all()
    assert np.array_equal(data['hands'], before_hands)
    assert np.array_equal(data['pose'], before_pose)


@pytest.mark.parametrize('problem', ['long_gap','trailing','wrong_hand','outside','fast_anchors'])
def test_rejects_unreliable_or_non_bracketed_repairs(problem):
    if problem=='long_gap':
        data=scene([True,False,False,False,True])
    elif problem=='trailing':
        data=scene([True,False,False])
    else:
        data=scene([True,False,True])
    missing=~data['observed']
    data['pose'][missing, 15, 0]=.1
    data['pose'][missing, 17, 0]=.1
    data['pose'][missing, 19, 0]=.1
    data['pose'][missing, 21, 0]=.1
    if problem=='wrong_hand':
        data['hands'][0,:,0]=.9
    if problem=='outside':
        data['hands'][0,20,0]=-.1
    if problem=='fast_anchors':
        data['hands'][2,:,0]+=.3
        data['pose'][2,[15,17,19,21],0]=.7
    source, _, _=fallback_geometry(**data)
    assert not (source==3).any()


def test_rejects_unknown_policy_parameters():
    data=scene([True,False,True])
    data['center_repair']['typo_threshold']=1
    with pytest.raises(ValueError, match='policy or keys'):
        fallback_geometry(**data)


def test_sub_box_disagreement_does_not_replace_a_possible_curved_motion():
    data=scene([True,False,True])
    data['pose'][1,[15,17,19,21],0]=.29
    source,centers,_=fallback_geometry(**data)
    assert source[1]==1  # ~0.85 box side: old 0.75 threshold could damage this frame
    assert np.allclose(centers[1],[58,50])
