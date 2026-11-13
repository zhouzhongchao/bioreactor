"""Retrain the fixed deployed Fusion architecture with input groups masked.

Removed standardized features are fixed to zero in TRAIN, VAL and TEST. This
keeps all layer sizes, parameter counts and paired-seed initialization identical.
Existing training data, deployed checkpoints and GUI settings are read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/output/recordings/fiting data/20260923_062542_476_slow_rom_gradual'
FEATURES = ['P_kPa', 'L_mm', 'd_active_mm', 'd_passive_mm',
            'dP_dt_kPa_s', 'dL_dt_mm_s']
VARIANTS = {
    'full_36D': list(range(6)),
    'without_pressure': [1, 2, 3, 4, 5],
    'without_length': [0, 2, 3, 4, 5],
    'without_active_displacement': [0, 1, 3, 4, 5],
    'without_passive_displacement': [0, 1, 2, 4, 5],
    'without_pressure_rate': [0, 1, 2, 3, 5],
    'without_length_rate': [0, 1, 2, 3, 4],
    'pressure_and_length_only': [0, 1],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
                        default=DATA / 'input_only_ablation_20261028')
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[2025, 2026, 2027])
    parser.add_argument('--max-epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--dry-run', action='store_true')
    opts = parser.parse_args()
    sys.path.insert(0, str(DATA))
    import numpy as np
    import pandas as pd
    import torch
    import train_haw36_vib_multihead_fxyz as trainer
    from train_haw42_multi_encoder_fx import seed_everything

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA training device unavailable; no training started')
    torch.set_num_threads(4)
    original_class = trainer.HAW36VIBMultiHeadRegressor
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    initial_hashes = {}
    rows = []
    start = time.time()
    for variant in opts.variants:
        keep = VARIANTS[variant]
        mask = np.array([int(j in keep) for i in range(6) for j in range(6)], dtype=np.float32)
        class MaskedRegressor(original_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_buffer('input_feature_mask', torch.tensor(mask))

            def forward(self, x, sample=True):
                return super().forward(x * self.input_feature_mask, sample=sample)

        checks = {}
        for seed in opts.seeds:
            seed_everything(seed)
            model = MaskedRegressor()
            digest = hashlib.sha256()
            for name, param in model.named_parameters():
                digest.update(name.encode())
                digest.update(param.detach().cpu().numpy().tobytes())
            checks[str(seed)] = digest.hexdigest()
            if seed in initial_hashes:
                assert initial_hashes[seed] == digest.hexdigest(), 'Initialization changed'
            initial_hashes[seed] = digest.hexdigest()
            # Changing a masked input must have exactly zero inference effect.
            x = torch.zeros(2, 36)
            x[:, mask == 0] = 1234
            with torch.no_grad():
                assert torch.equal(model(x, sample=False)[0],
                                   model(torch.zeros_like(x), sample=False)[0])
        folder = opts.output_dir / variant
        folder.mkdir(exist_ok=True)
        protocol = {
            'variant': variant, 'active_features_per_channel': [FEATURES[j] for j in keep],
            'active_input_dimension': int(mask.sum()), 'network_input_dimension': 36,
            'input_mask': mask.tolist(), 'paired_initial_parameter_sha256': checks,
            'seeds': opts.seeds, 'max_epochs': opts.max_epochs, 'patience': opts.patience,
            'beta': 1e-4, 'batch_size': 4096, 'learning_rate': .003,
            'weight_decay': 2e-5, 'early_stopping': 'validation macro RMSE',
            'test_excluded_processes': [25], 'feature_mask_applied': 'train, validation, test',
            'removed_feature_fill': 'zero after training-set standardization (training mean)',
            'scope': 'six existing 36D channel feature groups only; no pose/state additions',
        }
        (folder / 'ablation_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
        if opts.dry_run:
            print('DRY RUN OK', variant, int(mask.sum()), checks, flush=True)
            continue
        trainer.HAW36VIBMultiHeadRegressor = MaskedRegressor
        original_argv = sys.argv
        sys.argv = ['train_haw36_vib_multihead_fxyz.py',
                    '--data-dir', str(DATA / 'haw60_self_attention_dataset'),
                    '--response-xlsx', str(DATA / 'response.xlsx'),
                    '--output-dir', str(folder), '--beta', '0.0001',
                    '--seeds'] + list(map(str, opts.seeds)) + [
                    '--max-epochs', str(opts.max_epochs), '--patience', str(opts.patience)]
        print('\nINPUT-ONLY ABLATION', variant, 'elapsed_s', round(time.time()-start), flush=True)
        try:
            trainer.main()
        finally:
            sys.argv = original_argv
            trainer.HAW36VIBMultiHeadRegressor = original_class
        summary = pd.read_csv(folder / 'summary.csv').iloc[0].to_dict()
        rows.append(dict(variant=variant, active_input_dim=int(mask.sum()), **summary))
        result = pd.DataFrame(rows)
        if rows[0]['variant'] == 'full_36D':
            for axis in ['fx', 'fy', 'fz', 'macro']:
                key = 'test_%s_rmse_mean_n' % axis
                result['delta_%s_rmse_n' % axis] = result[key] - rows[0][key]
        result.to_csv(opts.output_dir / 'input_ablation_summary.csv', index=False)
        (opts.output_dir / 'progress.json').write_text(json.dumps({
            'completed_variants': list(result['variant']), 'total_variants': len(opts.variants),
            'elapsed_s': time.time()-start,
        }, indent=2), encoding='utf8')
    print('DONE', str(opts.output_dir), flush=True)


if __name__ == '__main__':
    main()
