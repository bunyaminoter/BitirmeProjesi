"""
TİD (Türk İşaret Dili / AUTSL) Preprocessor — Colab & Local.

AUTSL videoları için MediaPipe çıkarımı ve .npz önbellek üretimi:
  pose_landmarks (T,33,3), left/right_hand_landmarks (T,21,3),
  left/right_hand_crops (T,160,160,3), label, num_frames

Hız için cache önce Colab yerel diskine yazılır, sonunda Drive'a yedeklenir.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.core.config import load_config
from src.preprocessing.feature_cache import FeatureCache
from src.preprocessing.hand_cropper import HandCropper
from src.preprocessing.mediapipe_extractor import MediaPipeLandmarkExtractor
from src.utils.logging import setup_logging, get_logger
from src.data.datasets.tid_dataset import TIDDataset

DRIVE_PROJECT = "/content/drive/MyDrive/BitirmeProjesi"
COLAB_PROJECT = "/content/BitirmeProjesi"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(project_root / "configs/experiment/autsl_10_mobilenet.yaml"))
    parser.add_argument("--cache_dir", type=str, default=None, help="Varsayılan: config cache_dir; Colab'da önce yerel disk")
    parser.add_argument("--overwrite", action="store_true", help="Var olan cache dosyalarının üzerine yaz")
    parser.add_argument("--limit", type=int, default=None, help="Split başına en fazla N video (hızlı test için)")
    parser.add_argument("--pilot_size", type=int, default=None,
                        help="Her splitten sınıf dengeli N video seç (ön işleme pilotu)")
    parser.add_argument("--video_ids_file", type=Path, default=None,
                        help="JSON split -> video ID listesi; aynı videolarda kontrollü pilot için")
    parser.add_argument("--no_drive_backup", action="store_true", help="Bitince Drive'a yedekleme yapma")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Official AUTSL root; can also be set with AUTSL_DATA_DIR")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=None,
                        help="Splits to preprocess (AUTSL default: all three)")
    return parser.parse_args()


def sample_frames(video_path: str, num_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, min(total_frames, num_frames), dtype=int).tolist()
    if total_frames < num_frames:
        indices = list(range(total_frames))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            cap.release()
            raise ValueError(f"Video frame {idx} could not be decoded: {video_path}")
    cap.release()
    return frames


def pilot_indices(dataset: TIDDataset, requested: int) -> list[int]:
    """Select a reproducible, class-balanced pilot spread across each class."""
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        by_class[sample.label].append(index)
    classes = sorted(by_class)
    count = min(requested, len(dataset.samples))
    if count < len(classes):
        classes = [classes[index] for index in np.linspace(0, len(classes) - 1, count, dtype=int)]
    base, remainder = divmod(count, len(classes))
    chosen: list[int] = []
    for position, class_id in enumerate(classes):
        candidates = by_class[class_id]
        take = min(len(candidates), base + (position < remainder))
        chosen.extend(candidates[index] for index in np.linspace(0, len(candidates) - 1, take, dtype=int))
    return sorted(chosen)


def main():
    args = parse_args()
    if sum(value is not None for value in (args.limit, args.pilot_size, args.video_ids_file)) > 1:
        raise ValueError("Use only one of --limit, --pilot_size, --video_ids_file")
    if args.pilot_size is not None and args.pilot_size < 1:
        raise ValueError("--pilot_size must be positive")
    chosen_ids = None
    if args.video_ids_file is not None:
        chosen_ids = json.loads(args.video_ids_file.read_text(encoding="utf-8"))
        if not isinstance(chosen_ids, dict) or any(
            split not in {"train", "val", "test"} or not isinstance(ids, list)
            or not ids or len(ids) != len(set(ids)) or not all(isinstance(i, str) for i in ids)
            for split, ids in chosen_ids.items()
        ):
            raise ValueError("--video_ids_file must map splits to nonempty, unique video ID lists")
    config = load_config(args.config)
    if config.dataset.cache_derivation is not None:
        raise ValueError("Derived cache configs must be built with repair_hand_gaps.py")
    if args.data_dir:
        config.dataset.data_dir = args.data_dir
        if config.dataset.name == "autsl":
            os.environ["AUTSL_DATA_DIR"] = args.data_dir
    setup_logging(level="INFO")
    log = get_logger("preprocess_tid")

    in_colab = os.path.exists("/content/drive")

    # --- Yol çözümleme ---
    if in_colab and config.dataset.name != "autsl":
        for base in ["/content/tid_10_classes", f"{DRIVE_PROJECT}/tid_10_classes"]:
            if os.path.exists(f"{base}/labels.csv"):
                config.dataset.annotation_file = f"{base}/labels.csv"
                config.dataset.data_dir = f"{base}/videos"
                config.dataset.video_dir = f"{base}/videos"
                break
        cache_dir = Path(args.cache_dir or f"{COLAB_PROJECT}/cache/tid_10_features")
        log.info("Ortam: Google Colab")
    elif in_colab:
        cache_dir = Path(args.cache_dir or f"{COLAB_PROJECT}/{config.dataset.cache_dir}")
        log.info("Ortam: Google Colab, resmi AUTSL")
    else:
        cache_dir = Path(args.cache_dir or project_root / config.dataset.cache_dir)
        log.info("Ortam: Local")

    cache_dir.mkdir(parents=True, exist_ok=True)
    if config.dataset.name == "autsl":
        manifest_path = cache_dir / "manifest.json"
        manifest = {
            "dataset": "autsl",
            "pipeline_version": 2,
            "num_frames": config.dataset.num_frames,
            "hand_image_size": list(config.model.hand_encoder.input_size),
            "num_classes": config.dataset.num_classes,
            "class_ids": config.dataset.class_ids,
        }
        if manifest_path.exists():
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if recorded != manifest:
                raise ValueError(
                    f"Cache settings differ from {manifest_path}; use a new cache directory "
                    "when frame count, crop size, or class count changes."
                )
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(f"Cache klasörü : {cache_dir}")
    log.info(f"Video klasörü : {config.dataset.data_dir}")
    log.info(f"Etiket dosyası: {config.dataset.annotation_file}")

    cache = FeatureCache(cache_dir=str(cache_dir), cache_format="npz", store_hand_crops=True)
    extractor = MediaPipeLandmarkExtractor(enable_face=False, model_complexity=1)
    extractor.ensure_required_models()
    cropper = HandCropper(output_size=tuple(config.model.hand_encoder.input_size), padding_ratio=0.2)

    total_processed = 0
    splits = args.splits or (["train", "val", "test"] if config.dataset.name == "autsl" else ["train", "val"])
    for split in splits:
        ds = TIDDataset(config.dataset, split=split)
        if chosen_ids is not None:
            if split not in chosen_ids:
                raise ValueError(f"No {split} list in {args.video_ids_file}")
            wanted = set(chosen_ids[split])
            indices = [i for i, sample in enumerate(ds.samples) if sample.video_id in wanted]
            if len(indices) != len(wanted):
                found = {ds.samples[i].video_id for i in indices}
                raise ValueError(f"Unknown {split} video IDs: {sorted(wanted - found)}")
        elif args.pilot_size is not None:
            indices = pilot_indices(ds, args.pilot_size)
        elif args.limit is not None:
            indices = list(range(min(args.limit, len(ds.samples))))
        else:
            indices = range(len(ds.samples))
        n = len(indices)
        log.info(f"[*] {split.upper()}: {n} video işlenecek ({len(ds.class_to_idx)} sınıf)")

        processed, skipped, failed = 0, 0, 0
        failed_records = []
        for i in tqdm(indices, desc=f"TİD {split}", leave=True):
            sample = ds.get_metadata(i)

            if not args.overwrite and cache.exists(sample.video_id, split=split if config.dataset.name == "autsl" else None):
                skipped += 1
                continue

            vid_path = ds.get_video_path(i)
            if not vid_path.exists():
                log.warning(f"Video bulunamadı: {vid_path}")
                failed += 1
                failed_records.append({"video_id": sample.video_id, "split": split, "reason": "not_found", "path": str(vid_path)})
                continue

            try:
                frames = sample_frames(str(vid_path), config.dataset.num_frames)
            except Exception as e:
                log.error(f"Video okuma hatası ({sample.video_id}): {e}")
                failed += 1
                failed_records.append({"video_id": sample.video_id, "split": split, "reason": "read_error", "error": str(e)})
                continue
            if not frames:
                failed += 1
                failed_records.append({"video_id": sample.video_id, "split": split, "reason": "empty_frames"})
                continue

            all_pose, all_left_hand, all_right_hand, all_left_crop, all_right_crop = [], [], [], [], []
            pose_detected, left_detected, right_detected = [], [], []
            last_valid_left, last_valid_right = None, None

            for frame in frames:
                res = extractor.extract(frame)
                pose_detected.append(res.pose is not None)
                left_detected.append(res.left_hand is not None)
                right_detected.append(res.right_hand is not None)
                all_pose.append(res.pose if res.pose is not None else np.zeros((33, 3), dtype=np.float32))
                all_left_hand.append(res.left_hand if res.left_hand is not None else np.zeros((21, 3), dtype=np.float32))
                all_right_hand.append(res.right_hand if res.right_hand is not None else np.zeros((21, 3), dtype=np.float32))

                lc, rc = cropper.crop_hands(frame, res)

                # Temporal carry-forward (el kaybolursa son geçerli kırpımı kullan)
                if res.left_hand is not None:
                    last_valid_left = lc.image
                    all_left_crop.append(lc.image)
                else:
                    all_left_crop.append(last_valid_left if last_valid_left is not None else lc.image)

                if res.right_hand is not None:
                    last_valid_right = rc.image
                    all_right_crop.append(rc.image)
                else:
                    all_right_crop.append(last_valid_right if last_valid_right is not None else rc.image)

            features = {
                "pose_landmarks": np.stack(all_pose, axis=0),
                "left_hand_landmarks": np.stack(all_left_hand, axis=0),
                "right_hand_landmarks": np.stack(all_right_hand, axis=0),
                "left_hand_crops": np.stack(all_left_crop, axis=0),
                "right_hand_crops": np.stack(all_right_crop, axis=0),
                "label": np.array(sample.label, dtype=np.int64),
                "num_frames": np.array(len(frames), dtype=np.int64),
                "pose_detected": np.asarray(pose_detected, dtype=np.bool_),
                "left_hand_detected": np.asarray(left_detected, dtype=np.bool_),
                "right_hand_detected": np.asarray(right_detected, dtype=np.bool_),
            }
            cache.save(sample.video_id, features, split=split if config.dataset.name == "autsl" else None)
            processed += 1

        total_processed += processed
        log.info(f"[{split.upper()}] {processed} işlendi, {skipped} cache'den atlandı, {failed} hatalı.")

        failure_report = {
            "split": split,
            "processed": processed,
            "skipped": skipped,
            "failed_count": failed,
            "failed_videos": failed_records,
        }
        fail_path = cache_dir / f"preprocessing_failures_{split}.json"
        fail_path.write_text(json.dumps(failure_report, ensure_ascii=False, indent=2), encoding="utf-8")

        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        fail_path_out = outputs_dir / f"{cache_dir.name}_preprocessing_failures_{split}.json"
        fail_path_out.write_text(json.dumps(failure_report, ensure_ascii=False, indent=2), encoding="utf-8")

        if failed_records:
            log.warning(f"⚠️ {len(failed_records)} hatalı video kaydedildi: {fail_path} ve {fail_path_out}")
        else:
            log.info(f"✅ {split.upper()} splitinde hata yok ({fail_path_out}).")

    extractor.close()

    # --- Drive yedeği (Colab kapansa bile cache kaybolmasın) ---
    if in_colab and not args.no_drive_backup:
        cache_name = cache_dir.name
        drive_cache = Path(f"{DRIVE_PROJECT}/cache/{cache_name}")
        if drive_cache.resolve() != cache_dir.resolve():
            log.info(f"Cache Drive'a yedekleniyor -> {drive_cache}")
            drive_cache.mkdir(parents=True, exist_ok=True)
            local_manifest = cache_dir / "manifest.json"
            drive_manifest = drive_cache / "manifest.json"
            if drive_manifest.exists() and local_manifest.read_bytes() != drive_manifest.read_bytes():
                raise ValueError(f"Drive cache manifest differs: {drive_manifest}")
            subprocess.run(["cp", "-rn", f"{cache_dir}/.", str(drive_cache)], check=True)
            missing_backup = [
                str(path.relative_to(cache_dir)) for path in cache_dir.rglob("*.npz")
                if not (drive_cache / path.relative_to(cache_dir)).is_file()
            ]
            if missing_backup:
                raise RuntimeError(f"Drive backup incomplete; first missing files: {missing_backup[:5]}")

    n_files = len(list(cache_dir.rglob("*.npz")))
    print("\n" + "=" * 60)
    print("TİD ÖN İŞLEME TAMAMLANDI")
    print(f"Cache: {cache_dir}  ({n_files} adet .npz)")
    print("=" * 60)


if __name__ == "__main__":
    main()
