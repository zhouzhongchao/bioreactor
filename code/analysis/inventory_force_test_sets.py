"""Read-only inventory of measured forces and required input availability."""
from pathlib import Path
import json
import numpy as np
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exports/force_external_test_20261028'
CHANNELS=['A1+5','A2+3','A4+8','A6+7','A9','A10']


def main():
    reports=[]
    for path in (ROOT/'data/output/recordings').rglob('*.xlsx'):
        if path.name.startswith('~$'):continue
        book=None
        try:
            book=load_workbook(path,read_only=True,data_only=True)
            sheet=book.worksheets[0];iterator=sheet.iter_rows(values_only=True)
            headers=list(next(iterator))
            forces=['F_X_N','F_Y_N','F_Z_N']
            if not all(h in headers for h in forces):continue
            force_columns=[headers.index(h) for h in forces]
            required=[h for label in CHANNELS for h in ['P_'+label+'_kPa','L_'+label+'_mm','d_active_'+label+'_mm','d_passive_'+label+'_mm']]
            missing=[h for h in required if h not in headers]
            inputs=[headers.index(h) for h in required if h in headers]
            tidx=next((headers.index(h) for h in ['response_time_s','time_s','timestamp_local'] if h in headers),None)
            force_values=[];complete_values=[];times=[];total=0
            for row in iterator:
                total+=1
                try:
                    f=np.array([float(row[k]) for k in force_columns])
                    if not np.isfinite(f).all():continue
                except (IndexError,TypeError,ValueError):continue
                force_values.append(f)
                if missing:continue
                try:
                    x=np.array([float(row[k]) for k in inputs])
                    if not np.isfinite(x).all():continue
                    t=float(row[tidx]) if tidx is not None else None
                except (IndexError,TypeError,ValueError):continue
                complete_values.append(f);times.append(t)
            values=np.array(complete_values if complete_values else force_values)
            report={'path':str(path),'rows':total,'force_valid_rows':len(force_values),
                'complete_36D_input_force_rows':len(complete_values),'missing_input_columns':missing}
            if len(values):
                report.update(force_min_N=values.min(axis=0).tolist(),force_max_N=values.max(axis=0).tolist(),
                    force_abs_gt_5_pct=(100*(np.abs(values)>5).mean(axis=0)).tolist())
            if times and times[0] is not None:report['time_range_s']=[min(times),max(times)]
            reports.append(report)
            print(json.dumps(report,ensure_ascii=False),flush=True)
        except Exception as exc:print('ERROR',str(path),str(exc),flush=True)
        finally:
            if book is not None:book.close()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'available_test_data.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':main()
