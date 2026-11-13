"""Evaluate the deployed Fusion ensemble on the existing fixed test split."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/output/recordings/fiting data/20260923_062542_476_slow_rom_gradual'
MODELS = ROOT / 'code/control/models/force_estimator_v1'
OUT = ROOT / 'data/exports/final_force_test_20261028'


def main():
    import numpy as np
    import pandas as pd
    import torch
    sys.path.insert(0, str(DATA))
    from train_haw36_vib_multihead_fxyz import HAW36VIBMultiHeadRegressor
    from train_haw36_shared_fusion_fxyz import load_response_targets, load_split, exclude_processes
    from train_haw42_multi_encoder_fx import resolve_feature_columns

    torch.set_num_threads(4)
    manifest = json.loads((MODELS / 'manifest.json').read_text(encoding='utf8'))
    seeds = manifest['fusion_seeds']
    columns = [c for group in resolve_feature_columns(DATA / 'haw60_self_attention_dataset')[1] for c in group]
    times, targets, _ = load_response_targets(DATA / 'response.xlsx')
    predictions = []
    individual = []
    fitted_models = []
    test = None
    for seed in seeds:
        checkpoint = torch.load(str(MODELS / 'fusion' / ('model_seed%d.pt' % seed)), map_location='cpu')
        means = np.array(checkpoint['target_means_n'])
        stds = np.array(checkpoint['target_stds_n'])
        test = exclude_processes(load_split(DATA / 'haw60_self_attention_dataset/test.csv',
                                 'test', columns, times, targets, means, stds), [25])
        model = HAW36VIBMultiHeadRegressor()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with torch.no_grad():
            p = np.concatenate([model(test.x[i:i+512], sample=False)[0].numpy()
                                for i in range(0, len(test.x), 512)]) * stds + means
        predictions.append(p)
        fitted_models.append((model, means, stds))
        individual.append({'seed': seed, 'rmse_n': np.sqrt(np.mean((p-test.y_n)**2, axis=0)).tolist()})
    prediction = np.mean(predictions, axis=0)
    errors = prediction - test.y_n
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    result = {
        'model_version': manifest['model_version'], 'ensemble_seeds': seeds,
        'test_samples': len(prediction), 'test_process_ids': sorted(test.metadata.process_id.unique().astype(int).tolist()),
        'excluded_test_processes': [25], 'axis_order': ['Fx', 'Fy', 'Fz'],
        'rmse_n': rmse.tolist(), 'mse_n2': np.mean(errors**2, axis=0).tolist(),
        'mae_n': np.mean(np.abs(errors), axis=0).tolist(), 'macro_rmse_n': float(rmse.mean()),
        'individual_models': individual,
        'scope': 'Existing test set was used in historical seed selection; exploratory evaluation, not a new untouched test.',
    }
    frame = test.metadata.copy()
    for axis, name in enumerate(['Fx', 'Fy', 'Fz']):
        frame[name+'_true_N'] = test.y_n[:,axis]
        frame[name+'_pred_N'] = prediction[:,axis]
        frame[name+'_error_N'] = errors[:,axis]
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT/'test_predictions.csv', index=False)
    (OUT/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2), flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from force_input_ablation import VARIANTS
    masked_results = []
    for variant, keep in VARIANTS.items():
        mask = torch.tensor([float(j in keep) for i in range(6) for j in range(6)])
        x = test.x * mask
        outputs = []
        for model, means, stds in fitted_models:
            with torch.no_grad():
                z = np.concatenate([model(x[i:i+512], sample=False)[0].numpy()
                                    for i in range(0, len(x), 512)])
            outputs.append(z * stds + means)
        error = np.mean(outputs, axis=0) - test.y_n
        value = np.sqrt(np.mean(error**2, axis=0))
        item = {'variant': variant, 'active_input_dimension': len(keep)*6,
                'fx_rmse_n': value[0], 'fy_rmse_n': value[1], 'fz_rmse_n': value[2],
                'macro_rmse_n': value.mean(), 'delta_fx_rmse_n': value[0]-rmse[0],
                'delta_fy_rmse_n': value[1]-rmse[1], 'delta_fz_rmse_n': value[2]-rmse[2],
                'delta_macro_rmse_n': value.mean()-rmse.mean()}
        masked_results.append(item)
        print('FIXED WEIGHTS INPUT MASK', item, flush=True)
    pd.DataFrame(masked_results).to_csv(OUT/'fixed_model_input_mask_test.csv', index=False)
    (OUT/'input_mask_scope.txt').write_text(
        'Test-time standardized-input zero masking of the FIXED deployed ensemble. '
        'This is input dependence/sensitivity, NOT retraining ablation or a causal proof. '
        'Zero is the training-set mean. Masking may create out-of-distribution inputs. '
        'All cases use the identical 3157 test rows (process 4/10/16, excludes 25). '
        'Repeated-seed retraining ablation is saved separately.', encoding='utf8')


if __name__ == '__main__':
    main()
