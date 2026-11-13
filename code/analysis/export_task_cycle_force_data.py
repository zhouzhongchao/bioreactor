"""Export a complete task recording with pose-reset-derived cycle boundaries."""
from pathlib import Path
import csv
import json
import argparse
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data/exports/expanded_force_validation_20261028'
OUT = ROOT / 'data/exports/task_cycles_force_20261028'
AXES = ['Fx', 'Fy', 'Fz']
DIMS = [12, 36, 42, 54, 60]


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def style(ws):
    ws.freeze_panes = 'F2'
    ws.sheet_view.showGridLines = False
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 34
    for cell in ws[1]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='126C7C')
        cell.alignment = Alignment(wrap_text=True, vertical='center')
        ws.column_dimensions[cell.column_letter].width = max(18, min(len(str(cell.value)) + 2, 28))
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = '0.000000'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trim-boundary-idle', action='store_true')
    options = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    features = [row for row in read(SRC / 'expanded_validation_features_targets.csv') if row['dataset_id'] == 'E05']
    t = np.array([float(row['response_time_s']) for row in features])
    pose = np.array([[float(row[key]) for key in ['X_deg', 'Y_deg', 'phi_deg']] for row in features])
    zero = np.all(np.abs(pose) <= 1e-8, axis=1)
    starts = np.r_[0, np.flatnonzero(zero[1:] & ~zero[:-1]) + 1]
    ends = np.r_[starts[1:], len(features)]
    assert len(starts) == 10 and np.all(ends - starts == 323)
    assert len(features) == 3230 and np.allclose(np.diff(t), .05)
    assert np.all(np.diff([int(row['row_index']) for row in features]) == 1)
    loop_ids = np.zeros(len(features), dtype=int)
    for number, (a, b) in enumerate(zip(starts, ends), 1):
        loop_ids[a:b] = number
    windows = []
    compact_time = np.full(len(features), np.nan)
    compact_local = np.full(len(features), np.nan)
    next_time = 0.
    for a, b in zip(starts, ends):
        lo, hi = int(a), int(b)
        if options.trim_boundary_idle:
            p = np.array([[float(row['ch%d_P_kPa' % ch]) for ch in range(1, 7)] for row in features[a:b]])
            q = pose[a:b]
            smooth_p = np.array([np.median(p[max(0, i-2):i+3], axis=0) for i in range(len(p))])
            moving = np.array([np.ptp(q[max(0, i-4):i+5], axis=0).max() > .4 for i in range(len(q))])
            active = np.flatnonzero((smooth_p.max(axis=1) > 5) | moving)
            assert len(active)
            lo = int(a + max(0, active[0] - 2))
            hi = int(a + min(b-a, active[-1] + 3))
        windows.append((lo, hi))
        compact_local[lo:hi] = t[lo:hi] - t[lo]
        compact_time[lo:hi] = next_time + compact_local[lo:hi]
        next_time = float(compact_time[hi-1]) + .05
    keep = np.isfinite(compact_time)
    kept_n = int(keep.sum())
    duration = float(compact_time[keep][-1])
    wb = Workbook()
    summary = wb.active
    summary.title = 'Summary'
    summary.append(['Input dimensions', 'Samples', 'Task cycles inferred from pose reset',
                    'Duration s', 'Fx RMSE N', 'Fy RMSE N', 'Fz RMSE N', 'All axis RMSE <=4 N'])
    loops = wb.create_sheet('Cycles')
    loops.append(['Cycle', 'Start s', 'End s', 'Samples', 'First source row', 'Last source row',
                  'Fx RMSE N 36D', 'Fy RMSE N 36D', 'Fz RMSE N 36D', 'Original start s',
                  'Original end s', 'Removed prefix samples', 'Removed suffix samples'])
    all_metrics = {}
    for dim in DIMS:
        pred = read(SRC / ('E05_%dD_predictions.csv' % dim))
        assert len(pred) == len(features)
        ws = wb.create_sheet(str(dim))
        header = ['time_s', 'cycle_id', 'cycle_time_s', 'row_index', 'original_time_s']
        for axis in AXES:
            header += [axis + '_true_N', axis + '_pred_N', axis + '_variance_N2', axis + '_error_N']
        ws.append(header)
        errors = np.array([[float(row[axis + '_error_N']) for axis in AXES] for row in pred])
        for i, (row, source) in enumerate(zip(pred, features)):
            if not keep[i]:
                continue
            assert int(row['row_index']) == int(source['row_index'])
            assert np.isclose(float(row['response_time_s']), t[i])
            cycle_id = int(loop_ids[i])
            values = [float(compact_time[i]), cycle_id, float(compact_local[i]),
                      int(row['row_index']), float(t[i])]
            for axis in AXES:
                numbers = [float(row[axis + suffix]) for suffix in ['_true_N', '_pred_N', '_variance_N2', '_error_N']]
                assert np.isclose(numbers[0], float(source[axis + '_true_N']))
                assert np.isclose(numbers[1] - numbers[0], numbers[3])
                values += numbers
            ws.append(values)
        rmse = np.sqrt(np.mean(errors[keep] ** 2, axis=0)).tolist()
        all_metrics[str(dim)] = rmse
        summary.append([dim, kept_n, len(starts), duration, *rmse, bool(max(rmse) <= 4)])
        if dim == 36:
            for number, ((a, b), full_a, full_b) in enumerate(zip(windows, starts, ends), 1):
                value = np.sqrt(np.mean(errors[a:b] ** 2, axis=0)).tolist()
                loops.append([number, float(compact_time[a]), float(compact_time[b - 1]), int(b - a),
                              int(pred[a]['row_index']), int(pred[b - 1]['row_index']), *value,
                              float(t[a]), float(t[b-1]), int(a-full_a), int(full_b-b)])
    notes = wb.create_sheet('Read me')
    notes.append(['Provenance and interpretation'])
    for text in [
        'Source recording: ' + features[0]['source'],
        ('Boundary idle trimmed from the E05 task recording. Retain a single continuous interval per cycle, keeping all interior holds and direction changes. Compact time joins cycles at 0.05 s spacing; original_time_s retains acquisition time. No interpolation or alteration of true/predicted forces.' if options.trim_boundary_idle else
         'This is the complete E05 task recording, 3230 existing samples at 20 Hz, with its existing 0-to-161.45 s response time. No samples deleted, duplicated, interpolated or time-stretched by this export. No extra waiting intervals inserted.'),
        ('Boundary selection uses no prediction errors: active when any 5-sample-median pressure >5 kPa or any axis range >0.4 deg over 9 samples. Retain 2 extra samples (0.1 s) each side; remove only prefix/suffix outside that span.' if options.trim_boundary_idle else 'No boundary trimming.'),
        'Cycle boundaries are inferred from recorded simultaneous X/Y/phi pose resets to zero; there are 10 starts with exactly 323 samples between them. A command-level lap marker was not saved in this response workbook. These are task repetition boundaries, not slow-ROM pressure-cycle boundaries.',
        'All five dimensions use the same recorded sample IDs. Original measured forces and frozen ensemble predictions/variances remain unchanged. RMSE is recomputed on all rows and each task cycle.',
        'The 36D Fx RMSE on this export is %.6f N; the untrimmed original is 6.8616 N. Do not infer <=4 N accuracy from time compression.' % all_metrics['36'][0],
        'Prediction columns are means of seeds 2056/2059/2037; variance is pointwise population variance in N^2. Preserve within-cycle settling/hold phases, which are part of the recorded task.',
    ]:
        notes.append([text])
    for ws in wb:
        style(ws)
    notes.column_dimensions['A'].width = 140
    for row in notes.iter_rows(min_row=2):
        row[0].alignment = Alignment(wrap_text=True, vertical='center')
        notes.row_dimensions[row[0].row].height = 52
    path = OUT / ('task_10_cycles_force_predictions_idle_trimmed.xlsx' if options.trim_boundary_idle else 'task_10_cycles_force_predictions.xlsx')
    wb.save(path)
    check = load_workbook(path, read_only=True, data_only=True)
    for dim in map(str, DIMS):
        rows = list(check[dim].values)[1:]
        assert len(rows) == kept_n and np.allclose([row[0] for row in rows], compact_time[keep])
        assert np.all(np.diff([row[0] for row in rows]) > 0)
        actual = np.sqrt(np.mean(np.array([[row[i] for i in [8, 12, 16]] for row in rows]) ** 2, axis=0))
        assert np.allclose(actual, all_metrics[dim])
    check.close()
    (OUT / ('audit_idle_trimmed.json' if options.trim_boundary_idle else 'audit.json')).write_text(json.dumps(dict(samples=kept_n, cycles=len(starts),
         duration_s=duration, removed_samples=len(features)-kept_n, starts=starts.tolist(), retained_windows=windows,
         metrics=all_metrics, source=features[0]['source']), indent=2), encoding='utf-8')
    print(str(path))
    print(all_metrics['36'])


if __name__ == '__main__':
    main()
