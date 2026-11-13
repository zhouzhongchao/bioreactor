"""Export all eligible external-file evaluations without hiding hard cases."""
from pathlib import Path
import csv,json,math,argparse
import numpy as np
from openpyxl import Workbook,load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.comments import Comment

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exports/force_external_test_20261028'


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def header(sheet,values):
    cells=[]
    for text in values:
        c=WriteOnlyCell(sheet,value=text);c.fill=PatternFill('solid',fgColor='087F8C')
        c.font=Font(name='Arial',size=10,bold=True,color='FFFFFF')
        c.alignment=Alignment(wrap_text=True,horizontal='center',vertical='center');cells.append(c)
    sheet.row_dimensions[1].height=38;sheet.append(cells)


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,default=OUT)
    parser.add_argument('--only-dataset')
    parser.add_argument('--output-name',default='预测力_其他录制数据与Process25_逐条预测均值方差_20261028.xlsx')
    options=parser.parse_args()
    OUT=options.input_dir
    info=json.loads((OUT/'external_test_results.json').read_text())
    meta=info['datasets'];results=info['results']
    if options.only_dataset:
        assert options.only_dataset in meta
        meta={options.only_dataset:meta[options.only_dataset]}
        results=[r for r in results if r['dataset']==options.only_dataset]
    assert len(results)==5*len(meta)
    wb=Workbook(write_only=True)
    summary=wb.create_sheet('Summary');summary.freeze_panes='C2';summary.sheet_view.showGridLines=False
    h=['Dataset','Input dim','Samples','Fx RMSE N','Fy RMSE N','Fz RMSE N','Macro RMSE N',
       'Zero Fx RMSE N','Zero Fy RMSE N','Zero Fz RMSE N',
       'Fx RMSE when abs true >1N','Fy RMSE when abs true >1N','Fz RMSE when abs true >1N']
    header(summary,h)
    for result in results:
        values=[result['dataset'],result['input_dimension'],result['n'],result['fx_rmse_N'],result['fy_rmse_N'],result['fz_rmse_N'],result['macro_rmse_N']]
        values+=result['zero_predictor_rmse_N']+result['active_abs_gt_1N_rmse_N'];summary.append(values)
    summary.auto_filter.ref='A1:M%d'%(len(results)+1)
    from openpyxl.utils import get_column_letter
    for i in range(1,14):summary.column_dimensions[get_column_letter(i)].width=23
    index=wb.create_sheet('Datasets');index.sheet_view.showGridLines=False;index.freeze_panes='C2'
    h=['Dataset','Samples','Source recording or condition','Fx min N','Fx max N','Fy min N','Fy max N','Fz min N','Fz max N',
       'Fx abs<=1N %','Fy abs<=1N %','Fz abs<=1N %','Fx abs>5N %','Fy abs>5N %','Fz abs>5N %','Input provenance']
    header(index,h)
    for name,m in meta.items():
        values=[name,m['rows'],m['source']]
        for k in range(3):values += [m['force_min_N'][k],m['force_max_N'][k]]
        values+=m['abs_le_1N_pct']+m['abs_gt_5N_pct']
        values.append(m.get('input_provenance',m.get('input_reconstruction',{}).get('status','held-out condition in original recording' if name=='P25' else 'recorded complete inputs and measured forces')))
        source_cell=WriteOnlyCell(index,value=values[2]);source_cell.alignment=Alignment(wrap_text=True,vertical='center')
        values[2]=source_cell;index.row_dimensions[len(index.row_dimensions)+1].height=50
        index.append(values)
    index.column_dimensions['C'].width=95
    for i in list(range(1,3))+list(range(4,16)):index.column_dimensions[get_column_letter(i)].width=17
    index.column_dimensions['P'].width=50
    sums={};counts={}
    for dim in [12,36,42,54,60]:
        s=wb.create_sheet(str(dim));s.sheet_view.showGridLines=False;s.freeze_panes='F2'
        h=['time_s','dataset_id','process_id','row_index','response_time_s']
        for a in ['Fx','Fy','Fz']:h += [a+'_true_N',a+'_pred_N',a+'_variance_N2',a+'_error_N']
        header(s,h);sums[str(dim)]=np.zeros(3);count=0
        for name in meta:
            data=read(OUT/('%s_%dD_predictions.csv'%(name,dim)))
            assert len(data)==meta[name]['rows']
            truth=np.array([[float(r[a+'_true_N']) for a in ['Fx','Fy','Fz']] for r in data])
            pred=np.array([[float(r[a+'_pred_N']) for a in ['Fx','Fy','Fz']] for r in data])
            error=pred-truth
            item=next(r for r in results if r['dataset']==name and r['input_dimension']==dim)
            assert np.allclose(np.sqrt(np.mean(error**2,axis=0)),[item['fx_rmse_N'],item['fy_rmse_N'],item['fz_rmse_N']],atol=1e-8,rtol=1e-8)
            for row in data:
                values=[float(row['time_s']),name,int(row['process_id']),int(row['row_index']),float(row['response_time_s'])]
                for a in ['Fx','Fy','Fz']:values += [float(row[a+'_true_N']),float(row[a+'_pred_N']),float(row[a+'_variance_N2']),float(row[a+'_error_N'])]
                assert all(math.isfinite(v) for v in values if isinstance(v,(float,int)))
                s.append(values);count+=1
            sums[str(dim)]+=np.sum(error**2,axis=0)
        counts[str(dim)]=count;s.auto_filter.ref='A1:Q%d'%(count+1)
        for col in range(1,18):s.column_dimensions[get_column_letter(col)].width=22
    support_sheets=[]
    for filename,title in [('provided_vs_logged_reference.csv','Provided reference'),('alignment.csv','Input alignment')]:
        if not (OUT/filename).is_file():continue
        data=read(OUT/filename);support=wb.create_sheet(title);support_sheets.append(title)
        support.sheet_view.showGridLines=False;support.freeze_panes='B2';header(support,list(data[0]))
        for row in data:
            support.append([float(v) for v in row.values()])
        for i in range(1,len(data[0])+1):support.column_dimensions[get_column_letter(i)].width=28
    notes=wb.create_sheet('Protocol');notes.column_dimensions['A'].width=145
    for text in [
        'Frozen existing models, same seeds 2056/2059/2037, deterministic posterior means; predictions averaged and pointwise population variance retained (N^2).',
        info['protocol'],
        'No global pooled score is asserted: files may be correlated/overlapping. Report every recording separately.',
        'Summary includes a zero-output baseline and error restricted to samples with abs(measured force)>1 N for each axis; no targets altered or excluded to improve errors.',
        'time_s starts at zero within EACH dataset; use dataset_id to distinguish recordings. process_id=-1 means an external recording without original process labels.',
        *(['P25 is NOT a separate recording: it was excluded from the previously reported original test subset. It is a challenging held-out condition.'] if 'P25' in meta else []),
        *(['External E01..E06 files are not rows in the current training recording, but historical use for manual model tuning has not been audited; do not claim a new blinded validation.'] if any(name.startswith('E') for name in meta) else []),
        *[m['input_reconstruction']['zero_origin_limitation'] for m in meta.values() if 'input_reconstruction' in m],
        *[m['input_reconstruction']['matching'] for m in meta.values() if 'input_reconstruction' in m],
        *['原始力真值保持不变；部分位移输入由原始编码器日志重建。两份日志的位移归零起点约有1秒差异，五路分量零点不能完全核实，仅作重建输入补充测试。' for m in meta.values() if 'input_reconstruction' in m],
        *['Original response: '+m['input_reconstruction']['source_response']+'; raw encoders: '+m['input_reconstruction']['source_raw_encoders'] for m in meta.values() if 'input_reconstruction' in m],
    ]:
        c=WriteOnlyCell(notes,value=text);c.alignment=Alignment(wrap_text=True,vertical='center')
        notes.row_dimensions[len(notes.row_dimensions)+1].height=65
        notes.append([c])
    path=OUT/options.output_name;wb.save(path)
    check=load_workbook(path,read_only=True,data_only=True)
    assert check.sheetnames==['Summary','Datasets','12','36','42','54','60']+support_sheets+['Protocol']
    for dim in ['12','36','42','54','60']:
        it=check[dim].values;head=next(it);count=0;errors=np.zeros(3)
        for row in it:
            count+=1;errors+=np.array([row[8],row[12],row[16]])**2
        assert count==counts[dim] and np.allclose(errors,sums[dim],atol=1e-6,rtol=1e-9)
    check.close();print(path,flush=True);print(counts,flush=True)


if __name__=='__main__':main()
