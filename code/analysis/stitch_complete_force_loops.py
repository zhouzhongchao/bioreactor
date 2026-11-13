"""Compact complete pressure cycles from the previously delivered source sequences."""
from pathlib import Path
import csv
import json
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data/exports/expanded_force_validation_20261028'
PRIOR = ROOT / 'data/exports/qualified_complete_force_sequences_20261028/selection_audit.json'
OUT = ROOT / 'data/exports/stitched_complete_force_loops_20261028'
AXES = ['Fx', 'Fy', 'Fz']
DIMS = [12, 36, 42, 54, 60]


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def header(ws, fields):
    ws.append(fields)
    ws.freeze_panes = 'H2'
    ws.sheet_view.showGridLines = False
    ws.row_dimensions[1].height = 32
    for cell in ws[1]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='126C7C')
        cell.alignment = Alignment(wrap_text=True, vertical='center')
        ws.column_dimensions[cell.column_letter].width = max(17, min(len(str(cell.value)) + 2, 30))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(PRIOR.read_text())
    features = read(SRC / 'expanded_validation_features_targets.csv')
    loops, audits = [], []
    stitched_next = 0.
    for seg in prior['segments']:
        rows = [r for r in features if r['dataset_id'] == seg['dataset']
                and seg['start_s'] - 1e-6 <= float(r['response_time_s']) <= seg['end_s'] + 1e-6]
        t = np.array([float(r['response_time_s']) for r in rows])
        assert np.all(np.diff(t) > 0) and np.max(np.diff(t)) < 1
        p = np.array([[float(r['ch%d_P_kPa' % ch]) for ch in range(1, 7)] for r in rows])
        rest = np.max(p, axis=1) <= 10.
        changes = np.r_[0, np.flatnonzero(np.diff(rest.astype(int))) + 1, len(rows)]
        rests = [(int(a), int(b)) for a, b in zip(changes[:-1], changes[1:])
                 if rest[a] and t[b - 1] - t[a] >= .75]
        used = np.zeros(len(rows), dtype=bool)
        for (_, before_end), (after_start, _) in zip(rests[:-1], rests[1:]):
            if t[after_start] - t[before_end] < 3:
                continue
            lo = int(np.searchsorted(t, t[before_end] - 1))
            hi = int(np.searchsorted(t, t[after_start] + 1, side='right'))
            assert not np.any(used[lo:hi])
            chosen = rows[lo:hi]
            original_ids = [int(r['row_index']) for r in chosen]
            assert np.all(np.diff(original_ids) == 1)
            used[lo:hi] = True
            times = t[lo:hi]
            sample_dt = float(np.median(np.diff(times)))
            duration = float(times[-1] - times[0])
            loop = dict(loop_id='L%02d' % (len(loops) + 1), dataset=seg['dataset'],
                        process_id=int(chosen[0]['process_id']), source=seg['source'],
                        first_row=original_ids[0], last_row=original_ids[-1], samples=len(chosen),
                        original_start_s=float(times[0]), original_end_s=float(times[-1]),
                        time_start_s=stitched_next, time_end_s=stitched_next + duration,
                        duration_s=duration, boundary_dt_s=sample_dt, rows=chosen)
            stitched_next = loop['time_end_s'] + sample_dt
            loops.append(loop)
        audits.append(dict(dataset=seg['dataset'], process_id=int(rows[0]['process_id']),
                           source_rows=len(rows), retained_rows=int(used.sum()),
                           omitted_rows=int((~used).sum()),
                           reason='Long low-pressure rest outside complete cycles, or trailing cycle without >=0.75 s confirmed low-pressure return.'))
    assert len(loops) == 4

    wb = Workbook()
    summary = wb.active
    summary.title = 'Summary'
    header(summary, ['Input dimensions', 'Samples', 'Complete pressure loops', 'Display duration s',
                     'Fx RMSE N', 'Fy RMSE N', 'Fz RMSE N'])
    loop_ws = wb.create_sheet('Loops')
    header(loop_ws, ['Loop', 'Dataset', 'Process', 'Samples', 'Time start s', 'Time end s',
                     'Original start s', 'Original end s', 'First source row', 'Last source row',
                     'Fx RMSE N 36D', 'Fy RMSE N 36D', 'Fz RMSE N 36D', 'Source recording'])
    metrics = {}
    saved = {}
    for dim in DIMS:
        ws = wb.create_sheet(str(dim))
        fields = ['time_s', 'loop_id', 'loop_time_s', 'dataset_id', 'process_id', 'row_index', 'original_time_s']
        for axis in AXES:
            fields.extend([axis + '_true_N', axis + '_pred_N', axis + '_variance_N2', axis + '_error_N'])
        header(ws, fields)
        cache = {name: {int(r['row_index']): r for r in read(SRC / ('%s_%dD_predictions.csv' % (name, dim)))}
                 for name in {loop['dataset'] for loop in loops}}
        all_errors, exported = [], []
        for loop in loops:
            loop_errors = []
            for src_row in loop['rows']:
                row_id = int(src_row['row_index'])
                pred = cache[loop['dataset']][row_id]
                original_t = float(src_row['response_time_s'])
                local_t = original_t - loop['original_start_s']
                values = [loop['time_start_s'] + local_t, loop['loop_id'], local_t,
                          loop['dataset'], loop['process_id'], row_id, original_t]
                errors = []
                for axis in AXES:
                    vals = [float(pred[axis + suffix]) for suffix in ['_true_N', '_pred_N', '_variance_N2', '_error_N']]
                    assert np.isclose(vals[1] - vals[0], vals[3])
                    assert np.isclose(vals[0], float(src_row[axis + '_true_N']))
                    assert all(np.isfinite(vals)) and vals[2] >= 0
                    values.extend(vals)
                    errors.append(vals[3])
                ws.append(values)
                exported.append(values)
                all_errors.append(errors)
                loop_errors.append(errors)
            if dim == 36:
                loop['rmse_N'] = np.sqrt(np.mean(np.array(loop_errors) ** 2, axis=0)).tolist()
        rmse = np.sqrt(np.mean(np.array(all_errors) ** 2, axis=0)).tolist()
        metrics[str(dim)] = rmse
        summary.append([dim, len(exported), len(loops), loops[-1]['time_end_s']] + rmse)
        with (OUT / ('stitched_%dD.csv' % dim)).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(exported)
        saved[str(dim)] = exported
        ws.auto_filter.ref = ws.dimensions
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = '0.000000'
    for loop in loops:
        loop_ws.append([loop[k] for k in ['loop_id', 'dataset', 'process_id', 'samples', 'time_start_s',
                       'time_end_s', 'original_start_s', 'original_end_s', 'first_row', 'last_row']]
                       + loop['rmse_N'] + [loop['source']])
    loop_ws.column_dimensions['N'].width = 90
    for row in loop_ws.iter_rows(min_row=2):
        row[-1].alignment = Alignment(wrap_text=True)
        loop_ws.row_dimensions[row[0].row].height = 40
    audit_ws = wb.create_sheet('Source audit')
    audit_fields = list(audits[0])
    header(audit_ws, audit_fields)
    for audit in audits:
        audit_ws.append([audit[key] for key in audit_fields])
    audit_ws.column_dimensions['F'].width = 95
    for row in audit_ws.iter_rows(min_row=2):
        row[-1].alignment = Alignment(wrap_text=True)
        audit_ws.row_dimensions[row[0].row].height = 40
    notes = wb.create_sheet('Read me')
    header(notes, ['Method and provenance'])
    for note in [
        'time_s is a COMPACT DISPLAY TIME, starting at zero. Separate real experimental cycles are concatenated with one measured median sampling interval between them. It is not a continuous acquisition.',
        'original_time_s, dataset_id and row_index retain original acquisition provenance. loop_id marks every splice; loop_time_s starts at zero for each complete pressure cycle. No interpolation, duplication, resampling or change to forces/predictions.',
        'Scope: the three previously delivered complete source sequences. These sources had already been screened on 36D RMSE. This remains a performance-selected diagnostic example set, not an unbiased independent validation set.',
        'Cycle detection uses recorded pressures only: all six pressures <=10 kPa for >=0.75 s defines confirmed rest. A cycle must lie between two confirmed rests and have >=3 s active duration. Preserve about 1 s before activation and after low-pressure return.',
        'All complete cycles detected in those sources are included, even when an individual cycle RMSE exceeds 4 N. The unfinished final excursion in E02 lacks a confirmed return to low pressure and is excluded.',
        'Cycles are inferred pneumatic inflation/deflation excursions. No teaching-command or automatic-task loop labels were recorded in these workbooks; this does not establish that they are the same programmed robot task.',
        'All five model dimensions use identical source sample IDs and the same display time. Metrics are recomputed after rest removal. Derivative inputs and model predictions retain original causal evaluation history; they are not recomputed across artificial splices.',
    ]:
        notes.append([note])
        notes.cell(notes.max_row, 1).alignment = Alignment(wrap_text=True, vertical='center')
        notes.row_dimensions[notes.max_row].height = 50
    notes.column_dimensions['A'].width = 145
    path = OUT / 'complete_loops_compact_time.xlsx'
    wb.save(path)

    check = load_workbook(path, read_only=True, data_only=True)
    for dim, expected in saved.items():
        actual = list(check[dim].values)[1:]
        assert len(actual) == len(expected)
        assert np.allclose([r[0] for r in actual], [r[0] for r in expected])
        assert np.all(np.diff([r[0] for r in actual]) > 0)
        assert max(np.diff([r[0] for r in actual])) < .2
        assert np.allclose([[r[i] for i in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]] for r in actual],
                           [[r[i] for i in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]] for r in expected])
        rmse = np.sqrt(np.mean(np.array([[r[i] for i in [10, 14, 18]] for r in actual]) ** 2, axis=0))
        assert np.allclose(rmse, metrics[dim])
    check.close()
    report = dict(samples=len(saved['36']), loops=[{k: v for k, v in loop.items() if k != 'rows'} for loop in loops],
                  duration_s=loops[-1]['time_end_s'], metrics=metrics, source_audit=audits)
    (OUT / 'stitch_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(path=str(path), samples=report['samples'], loops=len(loops),
                          duration_s=report['duration_s'], rmse_N=metrics['36'])))


if __name__ == '__main__':
    main()
