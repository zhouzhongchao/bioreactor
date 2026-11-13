"""Export explicitly performance-selected continuous segments, not a blind test."""
from pathlib import Path
import csv
import json
import argparse
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data/exports/expanded_force_validation_20261028'
OUT = ROOT / 'data/exports/qualified_force_segments_20261028'
AXES = ['Fx', 'Fy', 'Fz']


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--complete-sequences', action='store_true')
    options = parser.parse_args()
    if options.complete_sequences:
        OUT = ROOT / 'data/exports/qualified_complete_force_sequences_20261028'
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((SRC / 'external_test_results.json').read_text())['datasets']
    segments = []
    audit = []
    for path in sorted(SRC.glob('*_36D_predictions.csv')):
        rows = read(path)
        t = np.array([float(r['response_time_s']) for r in rows])
        ids = np.array([int(r['row_index']) for r in rows])
        proc = np.array([int(r['process_id']) for r in rows])
        breaks = (np.diff(t) <= 0) | (np.diff(t) > 1) | (np.diff(ids) != 1) | (np.diff(proc) != 0)
        cuts = np.r_[0, np.flatnonzero(breaks) + 1, len(rows)]
        for start, stop in zip(cuts[:-1], cuts[1:]):
            blocks = (np.zeros(stop-start, dtype=int) if options.complete_sequences else
                      np.floor((t[start:stop] - t[start]) / 30).astype(int))
            for block in np.unique(blocks):
                ix = np.arange(start, stop)[blocks == block]
                if t[ix[-1]] - t[ix[0]] < 29:
                    continue
                errors = np.array([[float(rows[i][a + '_error_N']) for a in AXES] for i in ix])
                rmse = np.sqrt(np.mean(errors ** 2, axis=0))
                force = np.array([float(rows[i]['Fx_true_N']) for i in ix])
                audit.append([path.name.split('_36D')[0], int(proc[ix[0]]), float(t[ix[0]]),
                              float(t[ix[-1]]), len(ix), *rmse.tolist(), bool(np.max(rmse) <= 4)])
                if np.max(rmse) > 4 or np.ptp(force) < 10 or np.mean(np.abs(force) > 5) < .2:
                    continue
                segments.append(dict(segment='S%02d' % (len(segments) + 1),
                                     dataset=path.name.split('_36D')[0], indices=ix.tolist(),
                                     start_s=float(t[ix[0]]), end_s=float(t[ix[-1]]),
                                     n=len(ix), source=metadata[path.name.split('_36D')[0]]['source']))

    assert segments
    wb = Workbook()
    summary = wb.active
    summary.title = 'Summary'
    summary.append(['Segment', 'Dataset', 'Start response s', 'End response s', 'Samples',
                    'Input dimensions', 'Fx RMSE N', 'Fy RMSE N', 'Fz RMSE N'])
    pooled = {}
    for dim in [12, 36, 42, 54, 60]:
        ws = wb.create_sheet(str(dim))
        sample = read(SRC / (segments[0]['dataset'] + '_%dD_predictions.csv' % dim))[0]
        fields = list(sample)
        ws.append(['segment_id', 'dataset_id', 'segment_time_s'] + fields)
        errors_all = []
        for seg in segments:
            rows = read(SRC / (seg['dataset'] + '_%dD_predictions.csv' % dim))
            chosen = [rows[i] for i in seg['indices']]
            errors = np.array([[float(r[a + '_error_N']) for a in AXES] for r in chosen])
            rmse = np.sqrt(np.mean(errors ** 2, axis=0))
            if dim == 36:
                assert np.max(rmse) <= 4
            summary.append([seg['segment'], seg['dataset'], seg['start_s'], seg['end_s'],
                            seg['n'], dim] + rmse.tolist())
            errors_all.extend(errors)
            for row in chosen:
                values = [seg['source'] if k == 'source' else (int(row[k]) if k in ['row_index', 'process_id'] else float(row[k])) for k in fields]
                ws.append([seg['segment'], seg['dataset'], float(row['response_time_s']) - seg['start_s']] + values)
        pooled[str(dim)] = np.sqrt(np.mean(np.asarray(errors_all) ** 2, axis=0)).tolist()
        summary.append(['ALL SELECTED', '', '', '', len(errors_all), dim] + pooled[str(dim)])

    sources = wb.create_sheet('Sources')
    sources.append(['Segment', 'Dataset', 'Start s', 'End s', 'Samples', 'Original recording'])
    for seg in segments:
        sources.append([seg[k] for k in ['segment', 'dataset', 'start_s', 'end_s', 'n', 'source']])
    notes = wb.create_sheet('Selection')
    for value in [
        'Performance-selected examples; NOT an independent or unbiased validation set. Selection uses measured prediction error from the 36D model.',
        ('Evaluate each COMPLETE continuous recording or original labeled process. No time-window cropping or sample removal; all qualifying complete sequences are retained.'
         if options.complete_sequences else
         'Split each continuous recording/process into fixed non-overlapping 30 s windows from its start; retain windows spanning at least 29 s. No individual sample removal within a window.'),
        'All three 36D per-axis RMSE values must be <=4 N. Fx range must be >=10 N and at least 20% of samples must have abs(Fx)>5 N, to exclude idle-only examples.',
        'All five dimensions are evaluated on exactly the same selected sample IDs. No dimension-specific selection, retraining, target alteration or prediction alteration.',
        'Predictions are frozen three-seed means; variance is pointwise population variance N^2. Lower RMSE than 3 N is retained.',
        'Original denotes already reported held-out samples; E02 is another recorded file. The source workbook contains the complete unfiltered evaluation results.',
    ]:
        notes.append([value])
    audit_ws = wb.create_sheet('All sequence audit')
    audit_ws.append(['Dataset', 'Process', 'Start s', 'End s', 'Samples', 'Fx RMSE N',
                     'Fy RMSE N', 'Fz RMSE N', 'All axis RMSE <=4 N'])
    for row in audit:
        audit_ws.append(row)
    for ws in wb:
        ws.freeze_panes = 'C2'
        ws.sheet_view.showGridLines = False
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='126C7C')
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(38, max(18, len(str(col[0].value)) + 2))
    notes.column_dimensions['A'].width = 125
    for row in notes:
        row[0].alignment = Alignment(wrap_text=True, vertical='center')
        notes.row_dimensions[row[0].row].height = 48
    sources.column_dimensions['F'].width = 100
    path = OUT / ('qualified_complete_sequences.xlsx' if options.complete_sequences else 'qualified_continuous_segments.xlsx')
    wb.save(path)
    check = load_workbook(path, read_only=True, data_only=True)
    n = sum(s['n'] for s in segments)
    for dim in ['12', '36', '42', '54', '60']:
        assert sum(1 for _ in check[dim].values) == n + 1
    check.close()
    report = dict(samples=n, segments=segments, selected_36D_rmse_N=pooled['36'], all_dimensions_rmse_N=pooled)
    (OUT / 'selection_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(path=str(path), samples=n, segments=len(segments), rmse=pooled['36'])))


if __name__ == '__main__':
    main()
