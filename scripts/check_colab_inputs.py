"""Check paired cached AUTSL training inputs before spending GPU time.

Reads train/val only, does not create a model or start training. Optional
SHA256 verification detects incomplete/corrupt cache transfers.
"""
import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.core.config import load_config
from src.data.tid_datamodule import TIDDataModule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_config', type=Path, default=ROOT/'configs/experiment/autsl_10_base16_screen.yaml')
    parser.add_argument('--candidate_config', type=Path, default=ROOT/'configs/experiment/autsl_10_centerrepair_screen.yaml')
    parser.add_argument('--base_cache', type=Path)
    parser.add_argument('--candidate_cache', type=Path)
    parser.add_argument('--metadata_dir', type=Path)
    parser.add_argument('--checksum_manifest', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/colab_preflight.json')
    args = parser.parse_args()
    if args.metadata_dir:
        os.environ['AUTSL_DATA_DIR'] = str(args.metadata_dir.resolve())
    configs = [load_config(args.base_config), load_config(args.candidate_config)]
    for field in ('model', 'augmentation'):
        if asdict(getattr(configs[0],field)) != asdict(getattr(configs[1],field)):
            raise ValueError(f'Paired {field} settings differ')
    training = [asdict(c.training) for c in configs]
    for settings in training:
        settings.pop('checkpoint_dir')
    if training[0]!=training[1] or configs[0].seed!=configs[1].seed:
        raise ValueError('Paired training budget/seed differs')
    for field in ('class_ids','num_classes','num_frames'):
        if getattr(configs[0].dataset,field)!=getattr(configs[1].dataset,field):
            raise ValueError(f'Paired dataset {field} differs')
    checksums=json.loads(args.checksum_manifest.read_text(encoding='utf-8')) if args.checksum_manifest else None
    report=dict(training_started=False,paired_settings_equal=True,metadata_only_loader_verified=True,
                checksum_verification=bool(checksums),configs={},runtime={})
    for package in ('torch','torchvision','numpy','opencv-python','mediapipe'):
        try:
            report['runtime'][package]=version(package)
        except PackageNotFoundError:
            pass
    for role,c,override in zip(('base','candidate'),configs,(args.base_cache,args.candidate_cache)):
        cache=override or ROOT/c.dataset.cache_dir
        dm=TIDDataModule(c.dataset,c.training,cache_dir=cache,
            hand_image_size=tuple(c.model.hand_encoder.input_size),landmark_dim=c.model.landmark_encoder.input_dim)
        dm.setup(stage='fit')
        sample=dm._val_dataset[0]
        if sample['pose_landmarks'].shape!=(c.dataset.num_frames,225):
            raise ValueError('Unexpected landmark input shape')
        if sample['left_hand_images'].shape!=(c.dataset.num_frames,3,*c.model.hand_encoder.input_size):
            raise ValueError('Unexpected hand image shape')
        if not all(torch.isfinite(sample[k]).all() for k in ('pose_landmarks','left_hand_images','right_hand_images')):
            raise ValueError('Nonfinite validation input')
        if c.model.temporal.max_seq_len<c.dataset.num_frames:
            raise ValueError('max_seq_len is shorter than cache frames')
        if checksums:
            for relative,expected in checksums[role]['files'].items():
                path=cache/relative
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                    raise ValueError(f'Cache transfer checksum mismatch: {path}')
        report['configs'][c.name]=dict(train=len(dm._train_dataset),val=len(dm._val_dataset),
            cache=str(cache),seed=c.seed,epochs=c.training.epochs,
            hashes_checked=len(checksums[role]['files']) if checksums else 0,
            manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
