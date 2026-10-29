"""Raw-video verification of version-3 center repair, pilot or full train/val.

No selection means all official train/val videos. Test is never read.
All pixels/invariants are verified; full-run sheets focus on repaired centers
and unresolved crops. With a selection, every missing hand is rendered.
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
sys.path.insert(0, str(ROOT/'scripts'))
from preprocess_tid import sample_frames
from audit_wristcrop_pilot import card, inside, padding_fraction
from src.core.config import load_config
from src.core.types import LandmarkResult
from src.data.datasets.tid_dataset import TIDDataset
from src.preprocessing.hand_cropper import HandCropper
from src.preprocessing.wrist_fallback_crop import (
    center_repair_policy, crop_from_bounds, fallback_geometry,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT/'configs/experiment/autsl_10_mobilenet.yaml')
    source = ROOT/config.dataset.cache_dir
    manifest = json.loads((args.candidate/'manifest.json').read_text(encoding='utf-8'))
    deriv = manifest['derivation']
    if (deriv['method'] != 'pose_wrist_fallback_crop' or deriv['version'] != 3
            or deriv['source_manifest_sha256'] != hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()):
        raise ValueError('Unexpected version-3 source manifest')
    policy = center_repair_policy(deriv['center_repair'])
    scale = deriv.get('box_scale', 1.)
    selection = json.loads(args.selection.read_text(encoding='utf-8')) if args.selection else None
    cropper = HandCropper(tuple(manifest['hand_image_size']), padding_ratio=.2)
    records, cases, pages = [], [], []
    cells, page_cases = [], []

    def flush_page():
        if not cells:
            return
        name = f'centers_{len(pages)+1:03}.png'
        canvas = np.full((1040, 1200, 3), 245, np.uint8)
        cv2.putText(canvas, 'Orange=pose 1.5x | blue=repaired 1.5x | t=0-based | source=3 estimated center',
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .47, (20, 20, 20), 1)
        for j, cell in enumerate(cells):
            x, y = j%3*400, j//3*252+32
            canvas[y:y+252, x:x+400] = cell
            page_cases[j]['sheet'], page_cases[j]['cell'] = name, j+1
        if not cv2.imwrite(str(args.output_dir/name), canvas):
            raise OSError(name)
        pages.append(name)
        cells.clear()
        page_cases.clear()

    for split in ('train', 'val'):
        dataset = TIDDataset(config.dataset, split)
        indices = {s.video_id: i for i, s in enumerate(dataset.samples)}
        ids = selection[split] if selection else sorted(indices)
        if {p.stem for p in (args.candidate/split).glob('*.npz')} != set(ids):
            raise ValueError(f'{split} cache scope mismatch')
        for video_id in ids:
            sample = dataset.samples[indices[video_id]]
            frames = sample_frames(str(dataset.get_video_path(indices[video_id])), manifest['num_frames'])
            with np.load(source/split/f'{video_id}.npz') as a, np.load(args.candidate/split/f'{video_id}.npz') as b:
                base, candidate = dict(a), dict(b)
            for key in ('label', 'num_frames', 'pose_landmarks', 'pose_detected', 'left_hand_landmarks',
                        'right_hand_landmarks', 'left_hand_detected', 'right_hand_detected'):
                if not np.array_equal(base[key], candidate[key]):
                    raise ValueError(f'{video_id}: changed {key}')
            if int(candidate['label']) != sample.label or len(frames) != int(candidate['num_frames']):
                raise ValueError(f'{video_id}: metadata mismatch')
            record = dict(split=split, video_id=video_id, signer=sample.signer_id, label=sample.label,
                frames=len(frames), detected_checked=0, fallback_checked=0, center_repaired=0,
                changed_from_pose=0, unresolved=0, coverage_frames=0, pose_proxy_full=0,
                fallback_black=0, fallback_identical_previous=0, padding_gt_25pct=0,
                other_detected_center_inside=0)
            h, w = frames[0].shape[:2]
            for side, other in (('left', 'right'), ('right', 'left')):
                kwargs = dict(hands=base[f'{side}_hand_landmarks'], observed=base[f'{side}_hand_detected'],
                              pose=base['pose_landmarks'], pose_ok=base['pose_detected'], side=side,
                              width=w, height=h, box_scale=scale)
                pose_src, _, pose_boxes = fallback_geometry(**kwargs)
                src, centers, boxes = fallback_geometry(**kwargs, center_repair=policy)
                for key, expected in ((f'{side}_crop_source', src), (f'{side}_crop_center', centers),
                                      (f'{side}_crop_box', boxes), (f'{side}_crop_center_imputed', src==3)):
                    if not np.array_equal(candidate[key], expected):
                        raise ValueError(f'{video_id}: provenance mismatch {key}')
                images = candidate[f'{side}_hand_crops']
                for t, frame in enumerate(frames):
                    if src[t] == 0:
                        detected = LandmarkResult(**{f'{side}_hand': base[f'{side}_hand_landmarks'][t]})
                        expected = cropper.crop_hands(frame, detected)[0 if side=='left' else 1].image
                        if not (np.array_equal(images[t], expected) and np.array_equal(images[t], base[f'{side}_hand_crops'][t])):
                            raise ValueError(f'{video_id}/{side}/{t}: detected crop changed/misaligned')
                        record['detected_checked'] += 1
                        # Proxy is unchanged by a policy that only repairs missing hands.
                        if base['pose_detected'][t]:
                            from src.preprocessing.wrist_fallback_crop import video_box_side, pose_hand_center, crop_bounds
                            box = video_box_side(kwargs['hands'], kwargs['observed'], kwargs['pose'], kwargs['pose_ok'], side, w, h)
                            if box is not None:
                                bounds = crop_bounds(pose_hand_center(kwargs['pose'][t], side, w, h), max(box, 8.)*scale)
                                points = kwargs['hands'][t, :, :2]*[w, h]
                                record['coverage_frames'] += 1
                                record['pose_proxy_full'] += int(inside(points, bounds).all())
                        continue
                    old = (crop_from_bounds(frame, pose_boxes[t], manifest['hand_image_size'])
                           if pose_src[t]==1 else base[f'{side}_hand_crops'][t])
                    case = dict(split=split, video_id=video_id, label=sample.label, side=side, index=t,
                                source=int(src[t]), previous_source=int(pose_src[t]))
                    if src[t] in (1, 3):
                        expected = crop_from_bounds(frame, boxes[t], manifest['hand_image_size'])
                        if not np.array_equal(images[t], expected):
                            raise ValueError(f'{video_id}/{side}/{t}: fallback pixel mismatch')
                        record['fallback_checked'] += 1
                        record['center_repaired'] += int(src[t]==3)
                        record['changed_from_pose'] += int(not np.array_equal(old, images[t]))
                        record['fallback_black'] += int(not images[t].any())
                        record['fallback_identical_previous'] += int(t>0 and np.array_equal(images[t], images[t-1]))
                        case['padding'] = padding_fraction(boxes[t], w, h)
                        record['padding_gt_25pct'] += int(case['padding']>.25)
                        if base[f'{other}_hand_detected'][t]:
                            point = base[f'{other}_hand_landmarks'][t, :, :2].mean(axis=0)*[w, h]
                            case['other_detected_center_inside'] = bool(inside(point, boxes[t]))
                            record['other_detected_center_inside'] += int(case['other_detected_center_inside'])
                    else:
                        record['unresolved'] += 1
                        if not np.array_equal(images[t], base[f'{side}_hand_crops'][t]):
                            raise ValueError(f'{video_id}: unresolved crop changed')
                    if src[t]==3:
                        case['center'] = centers[t].tolist()
                        case['box'] = boxes[t].tolist()
                    if selection or src[t] in (2, 3):
                        cells.append(card(frame, old, images[t], pose_boxes[t] if pose_src[t]==1 else None,
                            boxes[t] if src[t] in (1,3) else None, case, base,
                            old_label='pose 1.5', new_label='center 1.5'))
                        page_cases.append(case)
                        if len(cells)==12:
                            flush_page()
                    cases.append(case)
            records.append(record)
            if len(records)%100==0:
                print(f'Verified {len(records)} videos', flush=True)
    flush_page()
    metrics = [k for k in records[0] if isinstance(records[0][k], int) and k!='label']
    summary = {k: sum(r[k] for r in records) for k in metrics}
    groups = {}
    for field in ('split', 'signer', 'label'):
        entries = defaultdict(list)
        for r in records:
            entries[str(r[field])].append(r)
        groups[field] = {key: dict(videos=len(rows), **{m: sum(r[m] for r in rows) for m in metrics})
                         for key, rows in entries.items()}
    report = dict(candidate=str(args.candidate), manifest=manifest, videos=len(records), summary=summary,
        groups=groups, records=records, missing_cases=cases, sheets=pages,
        visual_review='pending; sheets are not automatic acceptance',
        scope='fixed pilot' if selection else 'all official ten-class train/val; no test')
    (args.output_dir/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(videos=len(records), **summary)))


if __name__=='__main__':
    main()
