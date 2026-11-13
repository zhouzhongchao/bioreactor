"""Re-evaluate input ablations on the supplied force timeline, using raw sensors."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import hashlib
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from openpyxl import load_workbook
from evaluate_external_force_sets import features_from_record

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/output/recordings/fiting data/20260923_062542_476_slow_rom_gradual'
OUT = ROOT / 'data/exports/force_data_method_1_fin_ablation_20261028'
SEEDS = [2056, 2059, 2037]
AXES = ['Fx', 'Fy', 'Fz']


def read_book(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    rows = wb.worksheets[0].values
    header = next(rows)
    result = pd.DataFrame(list(rows), columns=header)
    wb.close()
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sys.path.insert(0, str(DATA))
    from train_haw36_vib_multihead_fxyz import HAW36VIBMultiHeadRegressor
    user_path = ROOT / 'force_data_method_1_fin.xlsx'
    force_path = next((ROOT / 'data/output/recordings').rglob('20260925_005704_134_force.xlsx'))
    response_path = force_path.with_name('20260925_005704_125_response.xlsx')
    user = read_book(user_path)
    logged = read_book(force_path)
    assert len(user) == len(logged) == 908
    y = user[['FX_real', 'FY_real', 'FZ_real']].to_numpy(dtype=float)
    logged_y = logged[['measured_FX_N', 'measured_FY_N', 'measured_FZ_N']].to_numpy(dtype=float)
    assert np.array_equal(y, logged_y)
    ft = user.force_time_s.to_numpy(dtype=float)
    assert np.array_equal(ft, logged.force_time_s.to_numpy(dtype=float))
    assert np.isfinite(y).all() and np.all(np.diff(ft) > 0)
    given_pred = user[['FX_pred', 'FY_pred', 'FZ__pred']].to_numpy(dtype=float)
    logged_pred = logged[['predicted_FX_N', 'predicted_FY_N', 'predicted_FZ_N']].to_numpy(dtype=float)
    audit = dict(user_file=str(user_path.relative_to(ROOT)), original_force_file=str(force_path.relative_to(ROOT)),
                 raw_sensor_file=str(response_path.relative_to(ROOT)), rows=len(user),
                 user_fx_equals_truth_plus_half_original_residual=bool(np.array_equal(given_pred[:, 0], y[:, 0]+.5*(logged_pred[:, 0]-y[:, 0]))),
                 user_fz_equals_truth_plus_half_original_residual=bool(np.array_equal(given_pred[:, 2], y[:, 2]+.5*(logged_pred[:, 2]-y[:, 2]))),
                 user_fy_equals_original=bool(np.array_equal(given_pred[:, 1], logged_pred[:, 1])),
                 user_given_rmse_N=np.sqrt(np.mean((given_pred-y)**2, axis=0)).tolist(),
                 historical_logged_rmse_N=np.sqrt(np.mean((logged_pred-y)**2, axis=0)).tolist(),
                 user_file_sha256=hashlib.sha256(user_path.read_bytes()).hexdigest())
    source = DATA / 'haw60_self_attention_dataset'
    names = pd.read_csv(source / 'feature_map.csv').sort_values('position_in_60D_0based').feature.tolist()
    scalers = pd.read_csv(source / 'train_scaler_stats.csv').set_index('feature')
    mu = scalers.loc[names, 'train_mean'].to_numpy(dtype=float)
    sigma = scalers.loc[names, 'train_std'].to_numpy(dtype=float)
    raw, response_y, response_meta, rejected = features_from_record(response_path, names, mu, sigma, return_raw=True)
    assert rejected == 0 and len(raw) == 902
    rt = np.array([r['response_time_s'] for r in response_meta])
    lookup = defaultdict(list)
    for i, force in enumerate(response_y):
        lookup[tuple(np.round(force, 6))].append(i)
    offsets = []
    for force_time, force in zip(ft, y):
        matches = lookup[tuple(np.round(force, 6))]
        if len(matches) == 1:
            offsets.append(float(rt[matches[0]] - force_time))
    assert len(offsets) >= 800
    offset = float(np.median(offsets))
    desired = ft + offset
    right = np.clip(np.searchsorted(rt, desired), 0, len(rt)-1)
    left = np.maximum(right-1, 0)
    indices = np.where(abs(rt[right]-desired) < abs(rt[left]-desired), right, left)
    alignment_error = rt[indices] - desired
    assert np.max(abs(alignment_error)) < .1
    assert np.all(np.diff(indices) >= 0)
    x = ((raw[indices]-mu)/sigma).astype(np.float32)
    audit['alignment'] = dict(method='Nearest recorded sensor snapshot to force timestamp after constant clock offset; no interpolation. Derivatives computed causally on original raw sensor stream before alignment.',
                              force_to_response_offset_s=offset, unique_force_clock_anchors=len(offsets),
                              max_abs_sensor_time_error_s=float(max(abs(alignment_error))),
                              median_abs_sensor_time_error_s=float(np.median(abs(alignment_error))),
                              distinct_sensor_snapshots=int(len(np.unique(indices))))
    provenance = pd.DataFrame(dict(force_row=np.arange(len(user)), force_time_s=ft,
                                   source_sensor_row=indices, sensor_time_s=rt[indices],
                                   estimated_response_time_s=desired, alignment_error_s=alignment_error))
    provenance.to_csv(OUT / 'alignment.csv', index=False)
    pd.DataFrame(raw[indices], columns=names).to_csv(OUT / 'aligned_raw_60D_inputs.csv', index=False)
    reference = pd.DataFrame(dict(time_s=ft))
    for k, axis in enumerate(AXES):
        reference[axis+'_true_N'] = y[:, k]
        reference[axis+'_user_provided_pred_N'] = given_pred[:, k]
        reference[axis+'_historical_logged_pred_N'] = logged_pred[:, k]
    reference.to_csv(OUT / 'provided_vs_logged_reference.csv', index=False)
    results = []
    for dim in [12, 36, 42, 54, 60]:
        folder = DATA / ('dimension_12_original_baseline_20261028' if dim == 12 else 'dimension_original_baseline_20261028') / ('%dD' % dim)
        cfg = json.loads((folder / 'input_config.json').read_text())
        selected = [names.index(name) for name in cfg['features_used']]
        width = cfg['channel_input_width']
        use_pose = cfg['pose_input_width'] > 0

        class InputRegressor(HAW36VIBMultiHeadRegressor):
            def __init__(self):
                super().__init__()
                if width != 6:
                    for encoder in self.encoders:
                        encoder.backbone[0] = nn.Linear(width, 64)
                if use_pose:
                    self.shared_fusion[0] = nn.Linear(234, 512)

            def forward(self, values):
                values = values[:, selected]
                latent = [encoder(values[:, i*width:(i+1)*width])[0] for i, encoder in enumerate(self.encoders)]
                if use_pose:
                    latent.append(values[:, -6:])
                shared = self.shared_fusion(torch.cat(latent, dim=1))
                return torch.cat([head(shared) for head in self.task_heads], dim=1)

        predictions = []
        for seed in SEEDS:
            ck = torch.load(str(folder / ('model_seed%d.pt' % seed)), map_location='cpu')
            model = InputRegressor()
            model.load_state_dict(ck['model_state_dict'])
            model.eval()
            with torch.no_grad():
                z = np.concatenate([model(torch.from_numpy(x[i:i+512])).numpy() for i in range(0, len(x), 512)])
            predictions.append(z*np.array(ck['target_stds_n'])+np.array(ck['target_means_n']))
        pred = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        error = pred-y
        frame = pd.DataFrame(dict(row_index=np.arange(len(user)), response_time_s=ft,
                                  process_id=np.full(len(user), -1), source='force_data_method_1_fin.xlsx', time_s=ft-ft[0]))
        for k, axis in enumerate(AXES):
            frame[axis+'_true_N'] = y[:, k]
            frame[axis+'_pred_N'] = pred[:, k]
            frame[axis+'_variance_N2'] = variance[:, k]
            frame[axis+'_error_N'] = error[:, k]
        frame.to_csv(OUT / ('FIN_%dD_predictions.csv' % dim), index=False)
        rmse = np.sqrt(np.mean(error**2, axis=0))
        active = [float(np.sqrt(np.mean(error[abs(y[:, k])>1, k]**2))) if np.any(abs(y[:, k])>1) else None for k in range(3)]
        result = dict(dataset='FIN', input_dimension=dim, n=len(y), fx_rmse_N=float(rmse[0]),
                      fy_rmse_N=float(rmse[1]), fz_rmse_N=float(rmse[2]), macro_rmse_N=float(rmse.mean()),
                      zero_predictor_rmse_N=np.sqrt(np.mean(y**2, axis=0)).tolist(), active_abs_gt_1N_rmse_N=active)
        results.append(result)
        print(json.dumps(result), flush=True)
    meta = dict(source='force_data_method_1_fin.xlsx; raw input recording: '+str(response_path.relative_to(ROOT)), rows=len(y),
                force_min_N=y.min(axis=0).tolist(), force_max_N=y.max(axis=0).tolist(),
                abs_le_1N_pct=(100*(abs(y)<=1).mean(axis=0)).tolist(), abs_gt_5N_pct=(100*(abs(y)>5).mean(axis=0)).tolist(),
                input_provenance='Timestamp-aligned recorded sensors; source user file contains only force targets/predictions, not model inputs')
    protocol = ('All 908 supplied measured force rows retained. Recomputed frozen-checkpoint predictions for all five input dimensions on the same timestamp-aligned original sensor snapshots. '
                'No retraining or parameter selection. Original TRAIN input/target scalers only. '
                'Force-to-response clock offset %.9f s; maximum nearest-sensor alignment error %.6f s. '
                'User-provided FX/FZ prediction columns equal ground truth plus half the historical logged residual; they are preserved as supplied reference only and are NOT used as model ablation results. '
                'Historical seed selection used the original reported test split. These results are separate-recording diagnostics, not new blinded validation.') % (offset, max(abs(alignment_error)))
    (OUT / 'external_test_results.json').write_text(json.dumps(dict(datasets={'FIN': meta}, results=results, protocol=protocol), indent=2), encoding='utf-8')
    (OUT / 'input_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
