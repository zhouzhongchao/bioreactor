"""Recover missing component inputs by exact raw/response sample matching.

No force values are inferred. Component displacements are measured signed raw
encoder changes relative to the raw-log origin; group A1+5 uses its original
recorded component values. The different origin convention is retained in the
provenance and this remains an exploratory reconstructed-input test.
"""
from pathlib import Path
import csv,json,collections
import numpy as np
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exports/force_external_test_20261028'
RAW=ROOT/'data/output/control_logs/slow_rom_raw_20260923_053412.csv'
SOURCE=ROOT/'data/output/recordings/fiting data/20260923_051325_559_slow_rom/response.xlsx'
LABELS=['A1+5','A2+3','A4+8','A6+7','A9','A10'];ORDER=[0,2,1,3,4,5]
GROUPS=[(0,4),(1,2),(3,7),(5,6),(8,),(9,)]


def main():
    with RAW.open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
    w=load_workbook(SOURCE,read_only=True,data_only=True);it=w.worksheets[0].values
    h=list(next(it));rows=[dict(zip(h,r)) for r in it];w.close()
    def response_key(r):
        return tuple(round(float(r[k]),4) for k in ['X_deg','Y_deg','phi_deg'])+tuple(round(float(r['P_'+l+'_kPa']),2) for l in LABELS)+tuple(round(float(r['L_'+l+'_mm']),3) for l in LABELS)
    def raw_key(r):
        return tuple(round(float(r[k]),4) for k in ['x','y','phi'])+tuple(round(float(r['P%d_kpa'%i]),2) for i in ORDER)+tuple(round(float(r['L_channel%d_mm'%i]),3) for i in ORDER)
    lookup=collections.defaultdict(list)
    for i,r in enumerate(rows):lookup[response_key(r)].append(i)
    pairs=[]
    for j,r in enumerate(raw):
        for i in lookup.get(raw_key(r),[]):
            offset=float(rows[i]['response_time_s'])-float(r['t'])
            if abs(offset)<5:pairs.append((i,j,offset))
    center=collections.Counter(round(p[2],2) for p in pairs).most_common(1)[0][0]
    delta=float(np.median([p[2] for p in pairs if abs(p[2]-center)<.01]))
    matches={}
    for i,j,offset in pairs:
        if abs(offset-delta)>1e-5:continue
        if i not in matches or abs(offset-delta)<abs(matches[i][1]-delta):matches[i]=(j,offset)
    recovered=[]
    for i,(j,offset) in sorted(matches.items()):
        row=rows[i].copy();r=raw[j]
        # Pressure, total length and pose are already exact matched measurements.
        for label,group in zip(LABELS,GROUPS):
            if label=='A1+5':continue
            row['d_active_'+label+'_mm']=float(np.mean([float(r['d_sensor%d_mm'%(k+1)]) for k in group]))
            row['d_passive_'+label+'_mm']=float(np.mean([float(r['d_sensor%d_mm'%(k+11)]) for k in group]))
        row['raw_log_row_index']=j;row['source_response_row_index']=i
        recovered.append(row)
    assert len(recovered)>10000
    assert all(np.isfinite([float(r['F_X_N']),float(r['F_Y_N']),float(r['F_Z_N'])]).all() for r in recovered)
    OUT.mkdir(exist_ok=True)
    path=OUT/'early_rom_exact_matched_reconstructed_inputs.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(recovered[0]));writer.writeheader();writer.writerows(recovered)
    note={
        'source_response':str(SOURCE),'source_raw_encoders':str(RAW),'matched_rows':len(recovered),
        'matching':'Same pose (rounded 4 decimals), all six pressures (2 decimals), all six total lengths (3 decimals), plus common clock offset residual <=10 microseconds',
        'response_time_minus_raw_time_s':delta,'max_clock_residual_s':max(abs(v[1]-delta) for v in matches.values()),
        'inputs_recovered':'Five groups active/passive displacements are averages of signed measured encoder changes from RAW log origin; A1+5 retains original response component values.',
        'zero_origin_limitation':'Raw-log origin is about one second after response recording start. Other five groups initial total lengths agree, but equality of their individual active/passive zero references cannot be proved from total lengths alone. Do not treat this as fully original recorded-input external validation.',
        'no_target_fitting':'All Fx/Fy/Fz are original measured values, unchanged. No force information used for clock fitting or zero adjustment.',
        'status':'exploratory reconstructed-input additional test, not equivalent to six fully recorded-input cases',
    }
    path.with_suffix('.json').write_text(json.dumps(note,indent=2),encoding='utf8')
    p=OUT/'available_test_data.json';inventory=json.loads(p.read_text(encoding='utf8'))
    inventory=[r for r in inventory if r['path']!=str(path)]
    inventory.append({'path':str(path),'rows':len(recovered),'force_valid_rows':len(recovered),'complete_36D_input_force_rows':len(recovered),'missing_input_columns':[],
                      'input_reconstruction':note})
    p.write_text(json.dumps(inventory,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(note,indent=2),flush=True)


if __name__=='__main__':main()
