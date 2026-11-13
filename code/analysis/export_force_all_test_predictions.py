"""One sheet per input dimension, retaining every held-out prediction row."""
from pathlib import Path
import csv
import argparse
import json
import math
import numpy as np
from openpyxl import Workbook,load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.comments import Comment

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENT=ROOT/'data/output/recordings/fiting data/20260923_062542_476_slow_rom_gradual'
OUT=ROOT/'data/exports/force_input_ablation_all_predictions_20261028'
NAME='预测力_输入维度消融_12_36_42_54_60维_测试集完整结果_20261028.xlsx'
SEEDS=[2056,2059,2037]
AXES=['x','y','z']


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--aggregate',action='store_true',help='Show only mean prediction and population variance across seeds')
    aggregate=parser.parse_args().aggregate
    OUT.mkdir(parents=True,exist_ok=True)
    wb=Workbook(write_only=True)
    canonical_keys=None;canonical_truth=None;checks={}
    headers=['time_s','process_id','row_index','response_time_s','Fx_true_N','Fy_true_N','Fz_true_N']
    for label in ['seed2056','seed2059','seed2037','ensemble']:
        headers += ['F%s_pred_%s_N'%(a,label) for a in AXES]
        headers += ['F%s_error_%s_N'%(a,label) for a in AXES]
    assert len(headers)==31
    if aggregate:
        headers=['time_s','process_id','row_index','response_time_s']
        for a in AXES:headers+=['F%s_true_N'%a,'F%s_pred_N'%a,'F%s_variance_N2'%a,'F%s_error_N'%a]
    ncols=len(headers)
    for dim in [12,36,42,54,60]:
        run=EXPERIMENT/('dimension_12_original_baseline_20261028' if dim==12 else 'dimension_original_baseline_20261028')
        folder=run/('%dD'%dim)
        data={seed:read(folder/('test_predictions_seed%d.csv'%seed)) for seed in SEEDS}
        first=data[2056];n=len(first);assert n==3157
        keys=[(int(r['row_index']),int(r['process_id'])) for r in first]
        assert len(set(keys))==n and sorted(set(k[1] for k in keys))==[4,10,16]
        truth=np.array([[float(r['F%s_true_n'%a]) for a in AXES] for r in first])
        times=np.array([float(r.get('time',r.get('response_time_s'))) for r in first])
        assert np.all(np.diff(times)>0)
        if canonical_keys is None:canonical_keys=keys;canonical_truth=truth.copy()
        assert keys==canonical_keys and np.array_equal(truth,canonical_truth)
        predictions={}
        metrics={int(m['seed']):m for m in read(folder/'metrics.csv')}
        for seed in SEEDS:
            r=data[seed]
            assert [(int(v['row_index']),int(v['process_id'])) for v in r]==keys
            assert np.array_equal(truth,np.array([[float(v['F%s_true_n'%a]) for a in AXES] for v in r]))
            assert np.array_equal(times,np.array([float(v.get('time',v.get('response_time_s'))) for v in r]))
            pred=np.array([[float(v['F%s_pred_n'%a]) for a in AXES] for v in r])
            assert np.isfinite(pred).all()
            error=pred-truth
            assert np.allclose(error,np.array([[float(v['F%s_error_n'%a]) for a in AXES] for v in r]),atol=1e-9,rtol=1e-9)
            rmse=np.sqrt(np.mean(error**2,axis=0))
            assert np.allclose(rmse,[float(metrics[seed]['test_f%s_rmse_n'%a]) for a in AXES],atol=1e-6,rtol=1e-6)
            predictions[seed]=pred
        ensemble=np.mean([predictions[seed] for seed in SEEDS],axis=0)
        variance=np.var([predictions[seed] for seed in SEEDS],axis=0,ddof=0)
        assert np.isfinite(variance).all() and np.all(variance>=0)
        stored=read(folder/'ensemble_predictions.csv')
        assert [(int(v['row_index']),int(v['process_id'])) for v in stored]==keys
        assert np.allclose(ensemble,np.array([[float(v['F%s_pred_n'%a]) for a in AXES] for v in stored]),atol=1e-9,rtol=1e-9)
        s=wb.create_sheet(str(dim));s.freeze_panes='E2';s.sheet_view.showGridLines=False
        from openpyxl.utils import get_column_letter
        s.auto_filter.ref='A1:%s%d'%(get_column_letter(ncols),n+1)
        for col,width in [('A',20),('B',13),('C',15),('D',22)]:s.column_dimensions[col].width=width
        from openpyxl.utils import get_column_letter
        for col in range(5,ncols+1):s.column_dimensions[get_column_letter(col)].width=23
        s.row_dimensions[1].height=34
        header=[]
        for j,h in enumerate(headers):
            c=WriteOnlyCell(s,value=h)
            color='#526879' if j<4 else ('#087F8C' if j<7 else ['#375C91','#6A5898','#9C6C32','#07675C'][(j-7)//6])
            if aggregate:color='#526879' if j<4 else ['#087F8C','#375C91','#6A5898'][(j-4)//4]
            c.fill=PatternFill('solid',fgColor=color.lstrip('#'));c.font=Font(name='Arial',size=10,bold=True,color='FFFFFF')
            c.alignment=Alignment(wrap_text=True,vertical='center',horizontal='center')
            if j==0:c.comment=Comment('All 3157 held-out test rows, process 4/10/16; process 25 excluded. time_s = source response time minus the FIRST TEST row time, not training start. Original inter-process time gaps are retained.','Data provenance')
            if 'error' in h:c.comment=Comment('Error = prediction minus measured truth; unit N.','Data provenance')
            if 'ensemble' in h:c.comment=Comment('Ensemble prediction = arithmetic mean of predictions from seeds 2056, 2059 and 2037. It is not the mean of RMSE values.','Data provenance')
            if aggregate and '_pred_' in h:c.comment=Comment('One combined prediction: arithmetic mean of the three fixed models (2056/2059/2037). This is not a newly trained single-seed model. Unit N.','Data provenance')
            if 'variance' in h:c.comment=Comment('Pointwise population variance across the three model predictions: sum((prediction - mean_prediction)^2)/3. ddof=0; unit N^2. Model disagreement, not measurement noise or variance over time.','Data provenance')
            header.append(c)
        s.append(header)
        for i in range(n):
            values=[float(times[i]-times[0]),keys[i][1],keys[i][0],float(times[i])]+truth[i].tolist()
            for label in SEEDS+['ensemble']:
                pred=ensemble if label=='ensemble' else predictions[label]
                values += pred[i].tolist()+(pred[i]-truth[i]).tolist()
            if aggregate:
                values=[float(times[i]-times[0]),keys[i][1],keys[i][0],float(times[i])]
                for k in range(3):values += [float(truth[i,k]),float(ensemble[i,k]),float(variance[i,k]),float(ensemble[i,k]-truth[i,k])]
            assert len(values)==ncols and all(math.isfinite(float(v)) for v in values)
            cells=[]
            for j,value in enumerate(values):
                c=WriteOnlyCell(s,value=value)
                c.number_format='0' if j in [1,2] else ('0.000' if j in [0,3] else '0.000000')
                cells.append(c)
            s.append(cells)
        checks[str(dim)]={'rows':n,'source':str(folder),'ensemble_rmse_N':np.sqrt(np.mean((ensemble-truth)**2,axis=0)).tolist(),
                          'pointwise_variance_ddof':0,'pointwise_variance_unit':'N^2','variance_column_sums':variance.sum(axis=0).tolist()}
    wb.properties.title='预测力 输入维度消融 测试集逐条真值和预测结果'
    wb.properties.description='Sheets 12/36/42/54/60. Original 36D weights retained. Identical 3157 test rows. All three fixed-seed predictions, ensemble and signed errors. Existing test was historically used for seed selection.'
    if aggregate:wb.properties.description='Sheets 12/36/42/54/60. One mean prediction and pointwise population variance across three fixed models, plus truth and signed error. 3157 test rows per sheet. No individual seed columns. Original 36D weights retained; no new model trained. Existing test has historical selection exposure.'
    name=NAME.replace('完整结果','均值与方差') if aggregate else NAME
    path=OUT/name;wb.save(path)
    # Verify the saved cells, not just the input arrays.
    check=load_workbook(path,read_only=True,data_only=True)
    assert check.sheetnames==['12','36','42','54','60']
    for s in check:
        iterator=s.iter_rows(values_only=True);assert list(next(iterator))==headers
        count=0;sum_sq=np.zeros(3);sum_var=np.zeros(3)
        for row in iterator:
            assert len(row)==ncols and all(isinstance(v,(int,float)) and math.isfinite(v) for v in row)
            if count==0:assert row[0]==0
            if aggregate:
                error=np.array([row[7],row[11],row[15]])
                assert np.allclose(error,[row[5]-row[4],row[9]-row[8],row[13]-row[12]],atol=1e-9)
                v=np.array([row[6],row[10],row[14]]);assert np.all(v>=0)
                sum_var+=v;sum_sq+=error**2
            else:sum_sq += np.array(row[28:31])**2
            count+=1
        assert count==3157
        assert np.allclose(np.sqrt(sum_sq/count),checks[s.title]['ensemble_rmse_N'],atol=1e-9,rtol=1e-9)
        if aggregate:assert np.allclose(sum_var,checks[s.title]['variance_column_sums'],atol=1e-9,rtol=1e-9)
    check.close()
    (OUT/('provenance_mean_variance.json' if aggregate else 'provenance.json')).write_text(json.dumps(checks,indent=2),encoding='utf8')
    print(path,flush=True);print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
