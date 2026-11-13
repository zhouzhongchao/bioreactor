"""Summarize fixed-weight sensitivity separately from paired retraining."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'data/output/recordings/fiting data/20260923_062542_476_slow_rom_gradual/input_only_ablation_20261028'
OUT = ROOT / 'data/exports/force_input_ablation_20261028'
LABELS = {
    'without_pressure': 'Pressure P', 'without_length': 'Length L',
    'without_active_displacement': 'Active displacement',
    'without_passive_displacement': 'Passive displacement',
    'without_pressure_rate': 'Pressure rate dP/dt',
    'without_length_rate': 'Length rate dL/dt',
}


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    rows = read(RUN/'input_ablation_summary.csv')
    if len(rows) != 8:
        raise RuntimeError('Training is not complete; no final report written')
    baseline = {r['seed']: r for r in read(RUN/'full_36D/metrics.csv')}
    paired = []
    for row in rows:
        variant = row['variant']
        metrics = read(RUN/variant/'metrics.csv')
        assert len(metrics) == 3 and {m['seed'] for m in metrics} == set(baseline)
        for axis in ['fx', 'fy', 'fz', 'macro']:
            key = 'test_%s_rmse_n' % axis
            diff = np.array([float(m[key])-float(baseline[m['seed']][key]) for m in metrics])
            paired.append({'variant': variant, 'axis': axis,
                           'paired_mean_delta_rmse_n': float(diff.mean()),
                           'paired_std_delta_rmse_n': float(diff.std(ddof=0)),
                           'n_seeds': 3, 'seed_deltas_n': diff.tolist()})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'paired_differences.json').write_text(json.dumps(paired,indent=2),encoding='utf8')
    fixed = {r['variant']: r for r in read(ROOT/'data/exports/final_force_test_20261028/fixed_model_input_mask_test.csv')}
    for source, dest in [(RUN/'input_ablation_summary.csv', OUT/'retrained_input_ablation.csv'),
                         (ROOT/'data/exports/final_force_test_20261028/fixed_model_input_mask_test.csv',OUT/'fixed_model_input_mask_test.csv')]:
        dest.write_bytes(source.read_bytes())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), sharey=True)
    order = list(LABELS)
    yy = np.arange(len(order))
    values = np.array([float(fixed[v]['delta_macro_rmse_n']) for v in order])
    axes[0].barh(yy, values, color='#B86725', height=.62)
    values2 = [next(p for p in paired if p['variant']==v and p['axis']=='macro') for v in order]
    axes[1].barh(yy, [p['paired_mean_delta_rmse_n'] for p in values2],
                 xerr=[p['paired_std_delta_rmse_n'] for p in values2],
                 color='#087F8C', capsize=3, height=.62)
    axes[0].set_yticks(yy, [LABELS[v] for v in order]);axes[0].invert_yaxis()
    for axis, title in zip(axes, ['Fixed final ensemble: test-time masking',
                                  'Retrained fixed network: 3 paired seeds']):
        axis.set_title(title, fontsize=11, pad=14)
        axis.axvline(0, color='#555555', linewidth=.9)
        axis.set_xlabel('Change in Macro RMSE after removal (N)')
        axis.grid(axis='x', alpha=.18);axis.set_axisbelow(True)
    fig.suptitle('Input information ablation for force prediction',fontsize=15,y=.98)
    fig.text(.02,.025,'Positive = worse without this input. Retraining error bars: paired population SD across 3 seeds.\nExisting test processes 4/10/16; process 25 excluded. Historical test-informed selection limits generalization claims.',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.12,1,.94])
    fig.savefig(OUT/'input_ablation.png',dpi=220)
    plt.close(fig)
    report = {
        'training_variants': 8, 'seeds': [2025,2026,2027], 'trained_models': 24,
        'same_architecture_parameters': 338763,
        'removed_features': 'zeroed AFTER training-only standardization in train/val/test',
        'test_rows': 3157, 'test_processes': [4,10,16], 'excluded_processes': [25],
        'no_deployed_models_changed': True,
        'retrained_macro_removal_rank': sorted([p for p in paired if p['axis']=='macro' and p['variant'] in LABELS],key=lambda x:x['paired_mean_delta_rmse_n'],reverse=True),
        'warning': 'Input importance is conditional on remaining redundant features and this model/data split. Fixed-weight masking is sensitivity, not retraining ablation. Existing test was historically used for seed selection.',
    }
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'output':str(OUT),'rank':report['retrained_macro_removal_rank']},indent=2))


if __name__=='__main__':
    main()
