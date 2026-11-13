"""Export the completed fixed-capacity comparison, with paired raw results."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,PatternFill,Alignment

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'data/output/recordings/fiting data/20260923_062542_476_slow_rom_gradual/dimension_ablation_20261028'
OUT=ROOT/'data/exports/force_dimension_ablation_20261028'


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    summary=read(RUN/'summary.csv')
    assert [int(r['input_dimension']) for r in summary]==[36,42,54,60]
    raw=[];initial_hashes=None;data_rows=None
    for row in summary:
        dim=int(row['input_dimension']);folder=RUN/('%dD'%dim)
        mask=json.loads((folder/'input_mask.json').read_text())
        assert mask['active_dimension']==dim and sum(mask['mask'])==dim
        assert mask['parameter_count']==342987
        if initial_hashes is None:initial_hashes=mask['paired_initialization_sha256']
        assert initial_hashes==mask['paired_initialization_sha256']
        metrics=read(folder/'metrics.csv')
        assert sorted(int(m['seed']) for m in metrics)==[2025,2026,2027]
        for m in metrics:
            prediction=read(folder/('test_predictions_seed%s.csv'%m['seed']))
            assert len(prediction)==3157
            row_ids=[r['row_index'] for r in prediction]
            if data_rows is None:data_rows=row_ids
            assert row_ids==data_rows
            for short in ['x','y','z']:
                err=np.array([float(p['F%s_error_n'%short]) for p in prediction])
                actual=float(np.sqrt(np.mean(err**2)))
                assert abs(actual-float(m['test_f%s_rmse_n'%short]))<1e-6
            raw.append([dim,int(m['seed'])]+[float(m['test_%s_rmse_n'%a]) for a in ['fx','fy','fz','macro']]+
                       [float(m['best_val_score_n']),int(m['best_epoch'])])
        for axis in ['fx','fy','fz','macro']:
            values=np.array([float(m['test_%s_rmse_n'%axis]) for m in metrics])
            assert abs(values.mean()-float(row['test_%s_rmse_mean_n'%axis]))<1e-9
            assert abs(values.std(ddof=0)-float(row['test_%s_rmse_std_n'%axis]))<1e-9
    OUT.mkdir(parents=True,exist_ok=True)
    wb=Workbook();s=wb.active;s.title='Comparison';r=wb.create_sheet('Per seed')
    s.sheet_view.showGridLines=False;r.sheet_view.showGridLines=False
    s.merge_cells('A1:I1');s['A1']='Force prediction input dimension ablation'
    s['A1'].font=Font(name='Arial',size=17,bold=True)
    s.merge_cells('A3:I3');s['A3']='Test RMSE (N), mean and population SD across 3 paired seeds. Lower is better.'
    headers=['Input dim','Fx mean','Fx SD','Fy mean','Fy SD','Fz mean','Fz SD','Macro mean','Macro SD']
    for j,h in enumerate(headers,1):s.cell(5,j,h)
    for i,row in enumerate(summary,6):
        vals=[int(row['input_dimension'])]
        for axis in ['fx','fy','fz','macro']:
            vals.extend([float(row['test_%s_rmse_mean_n'%axis]),float(row['test_%s_rmse_std_n'%axis])])
        for j,v in enumerate(vals,1):s.cell(i,j,v)
    desc=[('36D','6 channels x [P, L, active displacement, passive displacement, dP/dt, dL/dt]'),
          ('42D','36D + X, Y, phi + three angular velocities'),
          ('54D','36D + six channels x [inflate/hold/deflate state, active displacement rate, passive displacement rate]'),
          ('60D','54D + X, Y, phi + three angular velocities')]
    for i,(label,text) in enumerate(desc,12):
        s.cell(i,1,label);s.merge_cells(start_row=i,start_column=2,end_row=i,end_column=9)
        s.cell(i,2,text);s.cell(i,2).alignment=Alignment(wrap_text=True,vertical='center');s.row_dimensions[i].height=34
    notes=[
        'Same 60-slot architecture and 342,987 parameters for all groups. Missing standardized features are zeroed in TRAIN, VAL and TEST.',
        'This 36D condition is a masked fixed-capacity comparison; it is not the original deployed 36D architecture or its selected-seed ensemble.',
        'Seeds: 2025 / 2026 / 2027. Train 18,032; validation 4,210; test 3,157 rows (process 4/10/16). Process 25 excluded.',
        'AdamW lr 0.003, batch 4096, maximum 80 epochs, patience 12; same validation Macro RMSE checkpoint selection; beta 0.0001.',
        'Existing test set has historical test-informed selection exposure. These are exploratory comparisons, not a new untouched test.',
        'All 12 checkpoints, per-seed training curves, predictions and input masks are retained in the dimension_ablation_20261028 experiment folder.'
    ]
    for i,text in enumerate(notes,18):
        s.merge_cells(start_row=i,start_column=1,end_row=i,end_column=9)
        s.cell(i,1,text);s.cell(i,1).alignment=Alignment(wrap_text=True,vertical='center');s.row_dimensions[i].height=32
    r.append(['Input dim','Seed','Fx RMSE (N)','Fy RMSE (N)','Fz RMSE (N)','Macro RMSE (N)','Best val Macro (N)','Best epoch'])
    for values in raw:r.append(values)
    for sheet,headrow in [(s,5),(r,1)]:
        sheet.freeze_panes='B6' if sheet==s else 'C2'
        for cell in sheet[headrow]:
            cell.fill=PatternFill('solid',fgColor='0B7884');cell.font=Font(name='Arial',bold=True,color='FFFFFF',size=10)
            cell.alignment=Alignment(wrap_text=True,horizontal='center',vertical='center')
        sheet.row_dimensions[headrow].height=30
        for col in 'ABCDEFGHI':sheet.column_dimensions[col].width=14
    for row in s.iter_rows(min_row=6,max_row=9,min_col=2,max_col=9):
        for c in row:c.number_format='0.000'
    for row in r.iter_rows(min_row=2,min_col=3,max_col=7):
        for c in row:c.number_format='0.000'
    r.auto_filter.ref='A1:H13';s.auto_filter.ref='A5:I9'
    path=OUT/'force_input_36_42_54_60_comparison.xlsx';wb.save(path)
    check=load_workbook(path,data_only=True,read_only=True)
    assert check['Comparison']['A9'].value==60 and check['Per seed'].max_row==13
    check.close()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,4,figsize=(14,4.4))
    for ax,axis in zip(axs,['fx','fy','fz','macro']):
        means=[float(x['test_%s_rmse_mean_n'%axis]) for x in summary]
        errs=[float(x['test_%s_rmse_std_n'%axis]) for x in summary]
        ax.bar(np.arange(4),means,yerr=errs,capsize=4,color=['#42848C','#4A70A8','#C38B3C','#835D9E'],width=.65)
        ax.set_xticks(np.arange(4),['36D','42D','54D','60D']);ax.set_title(axis.upper()+' RMSE')
        ax.set_ylabel('RMSE (N)');ax.set_ylim(0,max(np.array(means)+errs)*1.25)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for i,v in enumerate(means):ax.text(i,v+errs[i]+ax.get_ylim()[1]*.02,'%.3f'%v,ha='center',fontsize=9)
    fig.suptitle('Input dimension ablation: fixed network and three paired seeds',fontsize=14)
    fig.text(.02,.025,'Mean +/- population SD; same 342,987 parameters and validation selection rule. Existing test processes 4/10/16; P25 excluded.',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.08,1,.92]);fig.savefig(OUT/'force_input_dimension_comparison.png',dpi=220);plt.close(fig)
    (OUT/'summary.csv').write_bytes((RUN/'summary.csv').read_bytes())
    (OUT/'protocol.json').write_bytes((RUN/'protocol.json').read_bytes())
    print(path,flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
