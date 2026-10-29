"""Two known center failures: diagnostic only, never writes a training cache.

Compare pose-center crops with a linear box-center estimate bracketed by
detected hands at samples 4 and 7. This is not a general repair algorithm.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from preprocess_tid import sample_frames
from src.core.config import load_config
from src.data.datasets.tid_dataset import TIDDataset
from src.preprocessing.wrist_fallback_crop import crop_bounds, pose_hand_center, square_crop, video_box_side


def main():
    config = load_config(ROOT/'configs/experiment/autsl_10_mobilenet.yaml')
    dataset = TIDDataset(config.dataset, 'train')
    video_id = 'signer33_sample505_color'
    index = next(i for i, s in enumerate(dataset.samples) if s.video_id == video_id)
    frames = sample_frames(str(dataset.get_video_path(index)), 16)
    with np.load(ROOT/config.dataset.cache_dir/'train'/f'{video_id}.npz') as packed:
        data = dict(packed)
    height, width = frames[0].shape[:2]
    units = np.array([width, height])
    hands = data['right_hand_landmarks']
    anchors = [4, 7]
    anchor_centers, anchor_checks = [], []
    for t in anchors:
        if not data['right_hand_detected'][t]:
            raise ValueError('Anchor hand is missing')
        xy = hands[t, :, :2]
        own = float(np.linalg.norm(hands[t, 0, :2]-data['pose_landmarks'][t, 16, :2]))
        other = float(np.linalg.norm(hands[t, 0, :2]-data['pose_landmarks'][t, 15, :2]))
        center = (xy.min(axis=0)+xy.max(axis=0))/2*units
        anchor_centers.append(center)
        anchor_checks.append(dict(index=t, pose_available=bool(data['pose_detected'][t]),
            own_wrist_distance=own, other_wrist_distance=other,
            finite_in_frame=bool(np.isfinite(xy).all() and ((xy>=0)&(xy<=1)).all()),
            bbox_center_pixels=center.tolist()))
    box = video_box_side(hands, data['right_hand_detected'].astype(bool), data['pose_landmarks'],
                         data['pose_detected'].astype(bool), 'right', width, height)
    report = dict(video_id=video_id, anchors=anchor_checks, units='source image pixels',
                  status='two-frame diagnostic, not a validated general policy', cases=[])
    canvas = np.full((2*280+40, 720, 3), 245, np.uint8)
    cv2.putText(canvas, 'Raw + boxes | pose center 1.5x | linear center 1.5x (diagnostic)',
                (6, 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1)
    for row, t in enumerate((5, 6)):
        alpha = (t-anchors[0])/(anchors[1]-anchors[0])
        linear_center = (1-alpha)*anchor_centers[0]+alpha*anchor_centers[1]
        pose_center = pose_hand_center(data['pose_landmarks'][t], 'right', width, height)
        crops = [square_crop(frames[t], c, max(box, 8.)*1.5, (160, 160)) for c in (pose_center, linear_center)]
        context = frames[t].copy()
        for c, color in ((pose_center, (255, 140, 0)), (linear_center, (0, 170, 255))):
            x, y, side = crop_bounds(c, max(box, 8.)*1.5)
            cv2.rectangle(context, (x, y), (x+side-1, y+side-1), color, 2)
        for col, image in enumerate([context, *crops]):
            canvas[40+row*280:280+row*280, col*240:(col+1)*240] = cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (240, 240))
        cv2.putText(canvas, f't={t} (0-based), anchors=4,7', (6, 309+row*280),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1)
        report['cases'].append(dict(index=t, pose_center_pixels=pose_center.tolist(),
            linear_center_pixels=linear_center.tolist(), shift_pixels=float(np.linalg.norm(linear_center-pose_center))))
    out = ROOT/'outputs/wristcrop15_pilot_qa'
    if not cv2.imwrite(str(out/'center_diagnostic.png'), canvas):
        raise OSError('Could not write center diagnostic')
    (out/'center_diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
