"""Export comparisons explicitly anchored to the original deployed models."""
from pathlib import Path
import csv,json,hashlib,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,PatternFill,Alignment

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'data/output/recordings/fiting data/20260923_062542_476_slow_rom_gradual/dimension_original_baseline_20261028'
OUT=ROOT/'data/exports/force_dimension_original_baseline_20261028'


def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    global RUN,OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,default=RUN)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--dimensions',nargs='+',type=int,default=[36,42,54,60])
    options=parser.parse_args();RUN=options.run_dir;OUT=options.output_dir
    rows=read(RUN/'summary.csv');assert [int(r['input_dimension']) for r in rows]==options.dimensions
    protocol=json.loads((RUN/'protocol.json').read_text())
    for seed,digest in protocol['original_checkpoint_sha256'].items():
        for p in [ROOT/('code/control/models/force_estimator_v1/fusion/model_seed%s.pt'%seed),RUN/('36D/model_seed%s.pt'%seed)]:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    assert abs(float(rows[0]['seed2056_fx_rmse_n'])-4.1024307556)<1e-5
    assert abs(float(rows[0]['ensemble_fx_rmse_n'])-3.42564142)<1e-5
    raw=[];ids=None
    for row in rows:
        dim=int(row['input_dimension']);metrics=read(RUN/('%dD/metrics.csv'%dim))
        assert [int(m['seed']) for m in metrics]==[2056,2059,2037]
        for m in metrics:
            pred=read(RUN/('%dD/test_predictions_seed%s.csv'%(dim,m['seed'])))
            assert len(pred)==3157
            current=[p['row_index'] for p in pred]
            if ids is None:ids=current
            assert ids==current
            for name in ['x','y','z']:
                err=np.array([float(p['F%s_error_n'%name]) for p in pred])
                assert abs(np.sqrt(np.mean(err**2))-float(m['test_f%s_rmse_n'%name]))<1e-6
            raw.append([dim,int(m['seed'])]+[float(m['test_%s_rmse_n'%a]) for a in ['fx','fy','fz','macro']]+[int(m['best_epoch'])])
    OUT.mkdir(parents=True,exist_ok=True)
    wb=Workbook();wb.remove(wb.active)
    for sheet_name,prefix,title in [
        ('Seed2056','seed2056','Single model seed 2056 - original 36D baseline retained'),
        ('Ensemble','ensemble','Original fixed seeds 2056 / 2059 / 2037 - ensemble predictions')]:
        s=wb.create_sheet(sheet_name);s.merge_cells('A1:F1');s['A1']=title;s['A1'].font=Font(name='Arial',bold=True,size=14)
        s.append([]);s.append(['Input dim','Fx RMSE (N)','Fy RMSE (N)','Fz RMSE (N)','Macro RMSE (N)','Parameters'])
        for row in rows:s.append([int(row['input_dimension'])]+[float(row['%s_%s_rmse_n'%(prefix,a)]) for a in ['fx','fy','fz','macro']]+[int(row['parameter_count'])])
        descriptions={12:'12D: six channels x [pressure P, total length L] only. No displacement, rate, pose or angular velocity inputs.',
            36:'36D: original deployed checkpoints evaluated directly. No retraining, no replacement, no seed changes.',
            42:'42D: original 36D features + three angles + three angular velocities.',
            54:'54D: original 36D features + six channels x [state, active displacement rate, passive displacement rate].',
            60:'60D: 54D + three angles + three angular velocities.'}
        for i,text in enumerate([descriptions[d] for d in options.dimensions]+[
            'Only necessary input Linear dimensions change. Original hidden widths, VIB, Fusion, heads, hyperparameters and split retained.',
            'Parameter counts differ slightly with input dimensions. This is NOT the previous fixed-60-slot masked architecture.',
            'Test: 3157 rows, process 4/10/16; process 25 excluded. Original test-informed seed selection limits independent generalization claims.',
        ],10):
            s.merge_cells(start_row=i,start_column=1,end_row=i,end_column=6);s.cell(i,1,text)
            s.cell(i,1).alignment=Alignment(wrap_text=True,vertical='center');s.row_dimensions[i].height=34
        for c in 'ABCDEF':s.column_dimensions[c].width=20
        for row in s.iter_rows(min_row=4,max_row=7,min_col=2,max_col=5):
            for c in row:c.number_format='0.000'
        s.freeze_panes='B4'
    s=wb.create_sheet('Three-seed mean')
    s.append(['Input dim','Fx mean (N)','Fx SD','Fy mean (N)','Fy SD','Fz mean (N)','Fz SD','Macro mean (N)','Macro SD'])
    for row in rows:
        values=[int(row['input_dimension'])]
        for a in ['fx','fy','fz','macro']:values += [float(row['test_%s_rmse_mean_n'%a]),float(row['test_%s_rmse_std_n'%a])]
        s.append(values)
    s['A8']='Mean of individual model RMSEs; population SD across the three original seeds. This is NOT ensemble RMSE.'
    s.merge_cells('A8:I9');s['A8'].alignment=Alignment(wrap_text=True,vertical='center')
    s=wb.create_sheet('Per seed');s.append(['Input dim','Seed','Fx RMSE (N)','Fy RMSE (N)','Fz RMSE (N)','Macro RMSE (N)','Best epoch'])
    for row in raw:s.append(row)
    for sheet in wb:
        sheet.sheet_view.showGridLines=False
        header=3 if sheet.title in ['Seed2056','Ensemble'] else 1
        for c in sheet[header]:
            c.fill=PatternFill('solid',fgColor='0B7884');c.font=Font(name='Arial',size=10,bold=True,color='FFFFFF')
            c.alignment=Alignment(wrap_text=True,horizontal='center',vertical='center')
        sheet.row_dimensions[header].height=30
        if sheet.title in ['Three-seed mean','Per seed']:
            for col in 'ABCDEFGHI':sheet.column_dimensions[col].width=16
            for row in sheet.iter_rows(min_row=2,max_row=1+len(rows) if sheet.title=='Three-seed mean' else 1+3*len(rows),min_col=2 if sheet.title=='Three-seed mean' else 3,max_col=9 if sheet.title=='Three-seed mean' else 6):
                for c in row:c.number_format='0.000'
            sheet.freeze_panes='B2'
    path=OUT/'force_input_comparison_original_baseline.xlsx';wb.save(path)
    check=load_workbook(path,read_only=True,data_only=True)
    assert abs(check['Seed2056']['B4'].value-4.1024307556)<1e-5
    assert abs(check['Ensemble']['B4'].value-3.42564142)<1e-5
    assert check['Per seed'].max_row==1+3*len(rows);check.close()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for k,(prefix,label) in enumerate([('seed2056','Single model 2056'),('ensemble','Original three-seed ensemble')]):
        for ax,a in zip(axes[k],['fx','fy','fz']):
            vals=[float(r['%s_%s_rmse_n'%(prefix,a)]) for r in rows]
            colors={12:'#B67A32',36:'#36818A',42:'#5279A7',54:'#C28D3B',60:'#865B9E'}
            ax.bar(np.arange(len(rows)),vals,color=[colors[int(r['input_dimension'])] for r in rows],width=.65)
            ax.set_xticks(np.arange(len(rows)),['%sD'%r['input_dimension'] for r in rows]);ax.set_title(label+' - '+a.upper())
            ax.set_ylabel('RMSE (N)');ax.set_ylim(0,max(vals)*1.2);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,'%.3f'%v,ha='center',fontsize=10)
    fig.suptitle('Input dimension comparison preserving the original 36D baseline',fontsize=14)
    fig.text(.02,.02,'36D original weights unchanged; other input variants use the same original seeds and training recipe. Only necessary input-layer sizes vary. P25 excluded.',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(OUT/'comparison_original_baseline.png',dpi=220);plt.close(fig)
    for name in ['summary.csv','protocol.json']:(OUT/name).write_bytes((RUN/name).read_bytes())
    print(path,flush=True)
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
