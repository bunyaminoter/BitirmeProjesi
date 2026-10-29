"""Verify every selected crop against raw video and render every missing hand.

Coverage on detected hands is a proxy, not missing-hand ground truth.
Overlap/padding flags are review aids, not automatic error labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from preprocess_tid import sample_frames
from inspect_preprocessing import draw_edges, HAND_EDGES, ARM_EDGES
from src.core.config import load_config
from src.core.types import LandmarkResult
from src.data.datasets.tid_dataset import TIDDataset
from src.preprocessing.hand_cropper import HandCropper
from src.preprocessing.wrist_fallback_crop import (
    crop_bounds, pose_hand_center, square_crop, video_box_side,
)


def inside(points, bounds):
    x, y, side = bounds
    return ((points >= [x, y]) & (points < [x+side, y+side])).all(axis=-1)


def padding_fraction(bounds, width, height):
    x, y, side = bounds
    area = max(0, min(width, x+side)-max(0, x)) * max(0, min(height, y+side)-max(0, y))
    return 1-area/(side*side)


def card(frame, old, new, bounds_old, bounds_new, case, features, *, old_label='1.00x', new_label='1.50x'):
    canvas = np.full((252, 400, 3), 245, np.uint8)
    rgb = frame.copy()
    t = case['index']
    if features['pose_detected'][t]:
        draw_edges(rgb, features['pose_landmarks'][t], ARM_EDGES, (255, 220, 0))
    for side, color in (('left', (255, 80, 80)), ('right', (80, 255, 80))):
        if features[f'{side}_hand_detected'][t]:
            draw_edges(rgb, features[f'{side}_hand_landmarks'][t], HAND_EDGES, color)
    for bounds, color in ((bounds_old, (255, 140, 0)), (bounds_new, (0, 170, 255))):
        if bounds is not None:
            x, y, side = bounds
            cv2.rectangle(rgb, (x, y), (x+side-1, y+side-1), color, 2)
    context = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (200, 200))
    canvas[36:236, :200] = context
    for x, crop in ((202, old), (301, new)):
        canvas[54:150, x:x+96] = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), (96, 96))
    title = f"{case['split']} {case['video_id'].replace('_color','')} {case['side'][0].upper()} t={case['index']}"
    cv2.putText(canvas, title, (3, 18), cv2.FONT_HERSHEY_SIMPLEX, .39, (20, 20, 20), 1)
    cv2.putText(canvas, old_label, (205, 45), cv2.FONT_HERSHEY_SIMPLEX, .34, (0, 90, 160), 1)
    cv2.putText(canvas, new_label, (303, 45), cv2.FONT_HERSHEY_SIMPLEX, .34, (160, 90, 0), 1)
    lines = [f"class={case['label']} source={case['source']}",
             f"padding={case.get('padding', 0):.2f}",
             f"other hand center={int(case.get('other_detected_center_inside', False))}"]
    for row, line in enumerate(lines):
        cv2.putText(canvas, line, (204, 176+21*row), cv2.FONT_HERSHEY_SIMPLEX, .34, (20, 20, 20), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=ROOT/'configs/experiment/autsl_10_wristcrop_pilot25.json')
    parser.add_argument('--candidate', type=Path, default=ROOT/'cache/autsl_10_features_wristcrop15_pilot')
    parser.add_argument('--output_dir', type=Path, default=ROOT/'outputs/wristcrop15_pilot_qa')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    config = load_config(ROOT/'configs/experiment/autsl_10_mobilenet.yaml')
    source = ROOT/config.dataset.cache_dir
    reference = ROOT/'cache/autsl_10_features_wristcrop'
    manifest = json.loads((args.candidate/'manifest.json').read_text(encoding='utf-8'))
    derivation = manifest['derivation']
    if (derivation['method'] != 'pose_wrist_fallback_crop' or derivation['version'] != 2
            or derivation['box_scale'] != 1.5
            or derivation['source_manifest_sha256'] != hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()):
        raise ValueError('Unexpected candidate manifest')
    scale = derivation['box_scale']
    cropper = HandCropper(tuple(manifest['hand_image_size']), padding_ratio=.2)
    records, cells, cases = [], [], []
    for split in ('train', 'val'):
        dataset = TIDDataset(config.dataset, split)
        indices = {s.video_id: i for i, s in enumerate(dataset.samples)}
        if {p.stem for p in (args.candidate/split).glob('*.npz')} != set(selection[split]):
            raise ValueError(f'{split}: pilot selection differs from cache files')
        for video_id in selection[split]:
            index = indices[video_id]
            frames = sample_frames(str(dataset.get_video_path(index)), manifest['num_frames'])
            data = []
            for directory in (source, reference, args.candidate):
                with np.load(directory/split/f'{video_id}.npz', allow_pickle=False) as packed:
                    data.append(dict(packed))
            base, ref, candidate = data
            record = dict(split=split, video_id=video_id, signer=video_id.split('_')[0], label=dataset.samples[index].label,
                detected_checked=0, fallback_checked=0, unresolved=0, coverage_frames=0,
                base_proxy_full=0, candidate_proxy_full=0, padding_gt_25pct=0,
                other_detected_center_inside=0, fallback_black=0, fallback_identical_previous=0)
            for key in ('label', 'num_frames', 'pose_landmarks', 'pose_detected',
                        'left_hand_landmarks', 'right_hand_landmarks', 'left_hand_detected', 'right_hand_detected'):
                if not np.array_equal(base[key], candidate[key]):
                    raise ValueError(f'{video_id}: changed {key}')
            if int(candidate['label']) != record['label'] or len(frames) != int(candidate['num_frames']):
                raise ValueError(f'{video_id}: metadata mismatch')
            height, width = frames[0].shape[:2]
            for side, other in (('left', 'right'), ('right', 'left')):
                observed = base[f'{side}_hand_detected'].astype(bool)
                pose_ok = base['pose_detected'].astype(bool)
                box = video_box_side(base[f'{side}_hand_landmarks'], observed, base['pose_landmarks'], pose_ok, side, width, height)
                src = candidate[f'{side}_crop_source']
                expected_source = np.where(observed, 0, np.where(pose_ok & (box is not None), 1, 2))
                if not np.array_equal(src, expected_source) or not np.array_equal(src, ref[f'{side}_crop_source']):
                    raise ValueError(f'{video_id}: source mismatch')
                old, new = ref[f'{side}_hand_crops'], candidate[f'{side}_hand_crops']
                for t, frame in enumerate(frames):
                    bounds_old = bounds_new = None
                    if pose_ok[t] and box is not None:
                        center = pose_hand_center(base['pose_landmarks'][t], side, width, height)
                        bounds_old = crop_bounds(center, max(box, 8.))
                        bounds_new = crop_bounds(center, max(box, 8.)*scale)
                    if observed[t]:
                        result = LandmarkResult(**{f'{side}_hand': base[f'{side}_hand_landmarks'][t]})
                        expected = cropper.crop_hands(frame, result)[0 if side == 'left' else 1].image
                        if not np.array_equal(expected, new[t]) or not np.array_equal(new[t], base[f'{side}_hand_crops'][t]):
                            raise ValueError(f'{video_id}/{side}/{t}: detected crop changed or misaligned')
                        record['detected_checked'] += 1
                        if bounds_new is not None:
                            points = base[f'{side}_hand_landmarks'][t, :, :2]*[width, height]
                            record['coverage_frames'] += 1
                            record['base_proxy_full'] += int(inside(points, bounds_old).all())
                            record['candidate_proxy_full'] += int(inside(points, bounds_new).all())
                        continue
                    case = dict(split=split, video_id=video_id, side=side, index=t, label=record['label'], source=int(src[t]))
                    if src[t] == 1:
                        expected = square_crop(frame, center, max(box, 8.)*scale, manifest['hand_image_size'])
                        if not np.array_equal(expected, new[t]):
                            raise ValueError(f'{video_id}/{side}/{t}: fallback not from current frame')
                        record['fallback_checked'] += 1
                        case['padding'] = padding_fraction(bounds_new, width, height)
                        record['padding_gt_25pct'] += int(case['padding'] > .25)
                        record['fallback_black'] += int(not new[t].any())
                        record['fallback_identical_previous'] += int(t > 0 and np.array_equal(new[t], new[t-1]))
                        if base[f'{other}_hand_detected'][t]:
                            other_center = base[f'{other}_hand_landmarks'][t, :, :2].mean(axis=0)*[width, height]
                            case['other_detected_center_inside'] = bool(inside(other_center, bounds_new))
                            record['other_detected_center_inside'] += int(case['other_detected_center_inside'])
                    else:
                        record['unresolved'] += 1
                        if not np.array_equal(new[t], base[f'{side}_hand_crops'][t]):
                            raise ValueError(f'{video_id}: unresolved crop changed')
                    cases.append(case)
                    cells.append(card(frame, old[t], new[t], bounds_old, bounds_new, case, base))
            records.append(record)
    metrics = [k for k in records[0] if isinstance(records[0][k], int) and k != 'label']
    totals = {k: sum(r[k] for r in records) for k in metrics}
    pages = []
    for start in range(0, len(cells), 12):
        page = np.full((4*252+32, 3*400, 3), 245, np.uint8)
        title = f"Missing hand samples {start+1}-{min(start+12,len(cells))}/{len(cells)} | orange=1x blue=1.5x | t=0-based"
        cv2.putText(page, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1)
        for j, cell in enumerate(cells[start:start+12]):
            x, y = (j % 3)*400, (j//3)*252+32
            page[y:y+252, x:x+400] = cell
            cases[start+j]['sheet'] = f'missing_{start//12+1:02}.png'
            cases[start+j]['cell'] = j+1
        path = args.output_dir/f'missing_{start//12+1:02}.png'
        if not cv2.imwrite(str(path), page):
            raise OSError(path)
        pages.append(path.name)
    grouped = {}
    for field in ('split', 'signer', 'label'):
        groups = defaultdict(list)
        for r in records:
            groups[str(r[field])].append(r)
        grouped[field] = {k: {m: sum(r[m] for r in v) for m in metrics} for k, v in groups.items()}
    report = dict(selection=str(args.selection), selection_sha256=hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        candidate=str(args.candidate), manifest=manifest, videos=len(records), summary=totals,
        grouped=grouped, records=records, missing_cases=cases, sheets=pages,
        visual_review_status='pending; numeric checks do not imply visual acceptance')
    (args.output_dir/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(videos=len(records), **totals), ensure_ascii=False))


if __name__ == '__main__':
    main()
