#!/usr/bin/env python3
"""Resumable construct-appropriate flatness and CPP trajectory extraction."""
from __future__ import annotations

import argparse, hashlib, json, os, platform, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import parselmouth
from parselmouth.praat import call

TIMES=np.arange(-20,21,5,dtype=int); EPS=1e-12; NFFT=512; AUDIT_SEED=20260819
PFLOOR=75.0; PCEIL=500.0; STEP=.002; MAXF=5000.0; PREEMPH=50.0
TAVG=0.0; QAVG=.0005; TOL=.05; INTERP='Parabolic'; QSTART=.001; QEND=.05
TREND='Straight'; FIT='Robust slow'; V5_SEG=.040; TRAJ_SEG=.040
BASE_COLS=['token_id','file_id','utterance_id','partition','system','label','speaker_id','phone_A','phone_B','phone_pair','boundary_class','boundary_time','A_duration_ms','B_duration_ms','sample_rate','eligibility','failure_reason','audio_path']
OUT_COLS=BASE_COLS+['relative_time_ms','feature','raw_value','frame_start_sample','frame_end_sample','requested_center_sec','realized_center_sec','center_error_samples','native_frame_index','native_frame_time_local_sec','native_mapping_error_ms','cpp_segment_start_sec','cpp_segment_end_sec']

def atomic_tsv(df,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');df.to_csv(q,sep='\t',index=False,na_rep='');os.replace(q,p)

def params(out):
 rows=[
  ('operator_name','CPPS (PowerCepstrogram: Get CPPS)'),('implementation','time-translated local frozen-v5 CPPS operator'),('exact_call','Get CPPS, subtract trend=yes, time average=0, quefrency average=0.0005, 75, 500, 0.05, Parabolic, 0.001, 0.05, Straight, Robust slow'),('praat_version',parselmouth.PRAAT_VERSION),('parselmouth_version',parselmouth.__version__),
  ('pitch_floor_hz',PFLOOR),('pitch_ceiling_hz',PCEIL),('powercepstrogram_time_step_sec',STEP),('maximum_frequency_hz',MAXF),('pre_emphasis_from_hz',PREEMPH),
  ('subtract_trend_before_smoothing',True),('time_averaging_window_sec',TAVG),('quefrency_averaging_window_sec',QAVG),('peak_tolerance',TOL),('peak_interpolation',INTERP),
  ('trend_line_start_sec',QSTART),('trend_line_end_sec',QEND),('trend_type',TREND),('fit_method',FIT),('frozen_v5_rectangular_segment_sec',V5_SEG),
  ('trajectory_local_rectangular_segment_sec',TRAJ_SEG),('trajectory_target_grid_ms','-20,-15,-10,-5,0,5,10,15,20'),('native_mapping','exact by local translation: every 40-ms operator segment is reset to 0 s and has its sole native frame at 0.020 s'),
  ('raw_support_requirement','outer target ±20 ms plus frozen local operator half-segment 20 ms = boundary ±40 ms; no padding/clipping'),('powercepstrogram_frame_note','each frozen 40-ms rectangular operator input yields nx=1, x1=0.020 s under Praat 6.1.38')]
 tab=pd.DataFrame(rows,columns=['parameter','recovered_value']);atomic_tsv(tab,out/'audit/cpp_recovered_parameters.tsv')
 final=out.parent/'final_six_feature_trajectory/audit';final.mkdir(parents=True,exist_ok=True);atomic_tsv(tab,final/'cpp_recovered_operator.tsv')
 (final/'cpp_parameter_recovery.md').write_text("""# Frozen v5 CPPS operator recovery\n\nThe frozen code calls Praat `PowerCepstrogram: Get CPPS`; therefore the correct construct name is **CPPS**, not framewise CPP. It extracts a 40-ms rectangular Sound segment centred on the requested time, creates a PowerCepstrogram with pitch floor 75 Hz, time step 2 ms, maximum frequency 5000 Hz, and pre-emphasis from 50 Hz, then calls `Get CPPS` with trend subtraction enabled, no time averaging, 0.5-ms quefrency averaging, 75–500 Hz peak search, tolerance 0.05, parabolic interpolation, a 1–50 ms straight trend, and robust-slow fitting. On this 40-ms input Praat 6.1.38 returns one native frame at local time 20 ms; the scalar is thus a smoothed CPPS summary of that local operator. The trajectory applies this identical operator to nine time-translated 40-ms segments.\n""",encoding='utf-8')
 (final/'flatness_v5_definition_check.md').write_text("""# Frozen v5 flatness definition\n\nRecovered directly from `run_full_study_robustness.py::spectrum_features`. Each 15-ms raw frame is multiplied by `numpy.hanning`; a 512-point real FFT is taken; magnitude is `abs(rfft)` and power is magnitude squared. Every returned real-FFT bin is retained, including DC and Nyquist. With epsilon `1e-12`, `safe_power = power + epsilon` and flatness is exactly `exp(mean(log(safe_power))) / mean(safe_power)`. The new trajectory evaluates this unchanged operator separately at each of the nine frozen short-time frame centres; no A/B averaging is performed.\n""",encoding='utf-8')

def exact_cpp_table(pc):
 # This is the internal Get CPPS sequence in frozen Praat 6.1.38.
 flat=call(pc,'Subtract trend',QSTART,QEND,TREND,FIT)
 smooth=call(flat,'Smooth',TAVG,QAVG)
 return call(smooth,'To Table (peak prominence)',PFLOOR,PCEIL,TOL,INTERP,QSTART,QEND,TREND,FIT)

def pc_from_part(sound,start,end):
 part=call(sound,'Extract part',float(start),float(end),'Rectangular',1.0,False)
 return call(part,'To PowerCepstrogram',PFLOOR,STEP,MAXF,PREEMPH)

def scalar_v5(sound,boundary):
 pc=pc_from_part(sound,boundary-.020,boundary+.020)
 return float(call(pc,'Get CPPS',True,TAVG,QAVG,PFLOOR,PCEIL,TOL,INTERP,QSTART,QEND,TREND,FIT))

def trajectory_cpps(sound,boundary,direct_validation=False):
 vals=[];checks=[]
 for rel in TIMES:
  center=boundary+rel/1000;pc=pc_from_part(sound,center-.020,center+.020);nx=int(call(pc,'Get number of frames'));x1=float(call(pc,'Get time from frame number',1))
  if nx!=1 or abs(x1-.020)>1e-9:raise ValueError(f'local_native_geometry:{rel}:{nx}:{x1}')
  value=float(call(pc,'Get CPPS',True,TAVG,QAVG,PFLOOR,PCEIL,TOL,INTERP,QSTART,QEND,TREND,FIT))
  if not np.isfinite(value):raise ValueError(f'nonfinite_cpps:{rel}')
  if direct_validation:
   tab=exact_cpp_table(pc);manual=float(call(tab,'Get value',1,'cpp'));checks.append((int(rel),value,manual,abs(value-manual)))
  vals.append((int(rel),value,1,x1,0.0))
 return vals,checks

def audit_mode(root,out):
 final_audit=root/'rebuild_v6_trajectory/final_six_feature_trajectory/audit';manifest=pd.read_csv(final_audit/'audit_token_ids.tsv',sep='\t')
 elig=pd.read_csv(root/'rebuild_v6_trajectory/extracted/trajectory_eligible_tokens.tsv',sep='\t',low_memory=False)
 old=pd.read_csv(root/'rebuild_v5/features/rebuild_v5_boundary_features_20ms.tsv',sep='\t',usecols=['token_id','boundary_cpp_db','cpp_status'])
 x=elig.merge(old,on='token_id',how='left',validate='one_to_one');sample=manifest[['token_id']].merge(x,on='token_id',how='left',validate='one_to_one');rows=[];translated=[];direct=[]
 for i,r in enumerate(sample.itertuples(index=False),1):
  try:
   sound=parselmouth.Sound(str(r.audio_path)); reproduced=scalar_v5(sound,float(r.boundary_time)); traj,checks=trajectory_cpps(sound,float(r.boundary_time),direct_validation=(i<=60)); t0=next(v for rel,v,_,_,_ in traj if rel==0)
   rows.append(dict(token_id=r.token_id,file_id=r.file_id,partition=r.partition,system=r.system,v5_stored_cpp=float(r.boundary_cpp_db),v5_operator_reproduced_cpp=reproduced,trajectory_t0_cpp=t0,
    abs_discrepancy_v5_operator=abs(reproduced-float(r.boundary_cpp_db)),status='ok',reason=''))
   translated.append(dict(token_id=r.token_id,partition=r.partition,system=r.system,v5_operator_at_boundary=reproduced,translated_operator_O0=t0,absolute_difference=abs(t0-reproduced),operator_equivalent_by_construction=True,status='ok'))
   for rel,a,b,diff in checks:direct.append(dict(token_id=r.token_id,relative_time_ms=rel,get_cpps_value=a,independent_internal_sequence_value=b,absolute_difference=diff,status='ok'))
  except Exception as e: rows.append(dict(token_id=r.token_id,file_id=r.file_id,partition=r.partition,system=r.system,v5_stored_cpp=float(r.boundary_cpp_db),v5_operator_reproduced_cpp=np.nan,abs_discrepancy_v5_operator=np.nan,status='invalid',reason=f'{type(e).__name__}:{e}'))
  if i%25==0: print('audit',i,'/',len(sample),flush=True)
 d=pd.DataFrame(rows);atomic_tsv(d,final_audit/'cpp_v5_operator_reproduction.tsv');atomic_tsv(pd.DataFrame(translated),final_audit/'cpp_translated_operator_validation.tsv');atomic_tsv(pd.DataFrame(direct),final_audit/'cpp_direct_praat_validation.tsv')
 ok=d[d.status=='ok']; summary=pd.DataFrame([
  dict(comparison='exact frozen 40-ms CPPS operator vs stored v5',n=len(ok),invalid=int((d.status!='ok').sum()),maximum_absolute_discrepancy=ok.abs_discrepancy_v5_operator.max(),mean_absolute_discrepancy=ok.abs_discrepancy_v5_operator.mean(),median_absolute_discrepancy=ok.abs_discrepancy_v5_operator.median(),rmse=float(np.sqrt(np.mean((ok.v5_operator_reproduced_cpp-ok.v5_stored_cpp)**2))),correlation=ok[['v5_stored_cpp','v5_operator_reproduced_cpp']].corr().iloc[0,1])])
 atomic_tsv(summary,out/'audit/cpps_v5_reproduction_summary.tsv')
 tr=pd.DataFrame(translated);dv=pd.DataFrame(direct);mapping=pd.DataFrame([dict(relative_time_ms=int(t),operator_segment_start_relative_ms=int(t-20),operator_segment_end_relative_ms=int(t+20),local_powercepstrogram_nx=1,local_native_frame_time_sec=.020,native_timing_offset_ms=0.0,mapping='exact local translation; no interpolation') for t in TIMES]);atomic_tsv(mapping,final_audit/'cpp_time_grid_mapping.tsv')
 md=f"""# Frozen v5 CPPS reproduction summary\n\nThe correct operator name is **CPPS**: Praat `PowerCepstrogram: Get CPPS`. The frozen 40-ms rectangular local segment yields one native PowerCepstrogram frame at local time 0.020 s. The trajectory is the exact translated operator O(t), implemented by shifting that same 40-ms segment and unchanged call to each requested center.\n\n- Audit seed: {AUDIT_SEED}; fixed manifest tokens: {len(sample)}.\n- Exact-v5 reproduction valid: {len(ok)}/{len(sample)}.\n- Maximum / mean / median / RMSE difference: {summary.iloc[0].maximum_absolute_discrepancy:.3g} / {summary.iloc[0].mean_absolute_discrepancy:.3g} / {summary.iloc[0].median_absolute_discrepancy:.3g} / {summary.iloc[0].rmse:.3g} dB.\n- Correlation: {summary.iloc[0].correlation:.12f}.\n- Translated O(0) maximum difference from the reproduced boundary operator: {tr.absolute_difference.max():.3g} dB.\n- Direct `Get CPPS` versus independently reconstructed internal sequence maximum difference: {dv.absolute_difference.max():.3g} dB across {len(dv)} token-time checks.\n- Native timing offset: 0 ms; no nearest-frame mapping or interpolation.\n""";(final_audit/'cpp_v5_reproduction_summary.md').write_text(md,encoding='utf-8')
 if len(ok)!=len(sample) or summary.iloc[0].maximum_absolute_discrepancy>5.1e-7 or tr.absolute_difference.max()>1e-12 or dv.absolute_difference.max()>1e-12: raise RuntimeError('Frozen CPPS operator reproduction/translation gate failed')
 print(summary.to_json(orient='records',indent=2))

def common_row(r):
 d={k:getattr(r,k) for k in BASE_COLS if hasattr(r,k)};d['eligibility']=True;d['failure_reason']='';return d

def flatness_rows(q):
 rows=[];fails=[]
 for path,g in q.groupby('audio_path',sort=True):
  try:
   audio,sr=sf.read(path,dtype='float64',always_2d=False);audio=audio.mean(axis=1) if audio.ndim==2 else audio
   starts=g.frame_start_sample.to_numpy(int); ends=g.frame_end_sample.to_numpy(int); n=np.unique(ends-starts)
   if len(n)!=1: raise ValueError('variable_frame_length')
   frames=np.stack([audio[a:b] for a,b in zip(starts,ends)]); window=np.hanning(int(n[0])); power=np.abs(np.fft.rfft(frames*window,n=NFFT,axis=1))**2;safe=power+EPS
   values=np.exp(np.mean(np.log(safe),axis=1))/np.mean(safe,axis=1)
   for r,val in zip(g.itertuples(index=False),values):
    d=common_row(r);d.update(relative_time_ms=int(r.relative_time_ms),feature='flatness',raw_value=float(val),frame_start_sample=int(r.frame_start_sample),frame_end_sample=int(r.frame_end_sample),requested_center_sec=float(r.requested_center_sec),realized_center_sec=float(r.realized_center_sec),center_error_samples=float(r.center_error_samples),native_frame_index=np.nan,native_frame_time_local_sec=np.nan,native_mapping_error_ms=np.nan,cpp_segment_start_sec=np.nan,cpp_segment_end_sec=np.nan);rows.append(d)
  except Exception as e:
   for token in g.token_id.unique(): fails.append(dict(token_id=token,feature='flatness',failure_reason=f'{type(e).__name__}:{e}'))
 return rows,fails

def cpps_rows(tokens,cpp_ok):
 rows=[];fails=[];audits=[]
 for path,g in tokens.groupby('audio_path',sort=True):
  try: sound=parselmouth.Sound(str(path))
  except Exception as e:
   for r in g.itertuples(index=False): fails.append(dict(token_id=r.token_id,feature='cpps',failure_reason=f'audio:{type(e).__name__}:{e}'))
   continue
  for r in g.itertuples(index=False):
   if r.token_id not in cpp_ok: continue
   duration=float(sound.get_total_duration()); b=float(r.boundary_time)
   if b-.040 < -1e-12 or b+.040 > duration+1e-12:
    fails.append(dict(token_id=r.token_id,feature='cpps',failure_reason='incomplete_boundary_plus_or_minus_40ms_operator_support'));continue
   try:
    vals,_=trajectory_cpps(sound,b)
    for rel,val,idx,native,err in vals:
     center=b+rel/1000;d=common_row(r);d.update(relative_time_ms=rel,feature='cpps',raw_value=val,frame_start_sample=np.nan,frame_end_sample=np.nan,requested_center_sec=center,realized_center_sec=center,center_error_samples=np.nan,native_frame_index=idx,native_frame_time_local_sec=native,native_mapping_error_ms=err,cpp_segment_start_sec=center-.020,cpp_segment_end_sec=center+.020);rows.append(d)
    audits.append(dict(token_id=r.token_id,feature='cpps',native_frames_per_local_operator=1,native_frame_time_sec=.020,native_timing_offset_ms=0,eligibility=True,failure_reason=''))
   except Exception as e: fails.append(dict(token_id=r.token_id,feature='cpps',failure_reason=f'{type(e).__name__}:{e}'))
 return rows,fails,audits

def process_audio_group(q):
 """Independent per-audio worker; statistical/acoustic definitions are unchanged."""
 q=q.sort_values(['token_id','relative_time_ms']);tok=q.drop_duplicates('token_id');ok=set(tok.loc[tok.cpp_frozen_ok,'token_id'])
 fr,ff=flatness_rows(q);cr,cf,ca=cpps_rows(tok,ok);return fr+cr,ff+cf,ca

def full_mode(root,out,workers):
 source_dir=root/'rebuild_v6_trajectory/extracted/full_chunks';chunkdir=out/'extracted/chunks';chunkdir.mkdir(parents=True,exist_ok=True)
 old=pd.read_csv(root/'rebuild_v5/features/rebuild_v5_boundary_features_20ms.tsv',sep='\t',usecols=['token_id','cpp_status','boundary_cpp_db'])
 cpp_ok=set(old.loc[(old.cpp_status=='ok')&np.isfinite(old.boundary_cpp_db),'token_id'].astype(str));man=[]
 cols=['token_id','file_id','utterance_id','partition','system','label','speaker_id','phone_A','phone_B','phone_pair','boundary_class','boundary_time','A_duration_ms','B_duration_ms','sample_rate','eligibility','failure_reason','audio_path','relative_time_ms','frame_start_sample','frame_end_sample','requested_center_sec','realized_center_sec','center_error_samples','feature']
 for cp in sorted(source_dir.glob('chunk_*.parquet')):
  no=cp.stem.split('_')[-1];op=chunkdir/f'extension_{no}.parquet';fp=chunkdir/f'extension_{no}_failures.tsv';ap=chunkdir/f'extension_{no}_cpp_audit.tsv';done=chunkdir/f'extension_{no}.complete.json'
  if done.exists() and op.exists() and fp.exists():man.append(json.loads(done.read_text()));continue
  t=time.perf_counter();x=pd.read_parquet(cp,columns=cols);q=x[x.feature=='energy'].copy();q=q.sort_values(['audio_path','token_id','relative_time_ms']);q['cpp_frozen_ok']=q.token_id.isin(cpp_ok);tok=q.drop_duplicates('token_id')
  groups=[g.copy() for _,g in q.groupby('audio_path',sort=True)];rr=[];ff=[];ca=[]
  with ProcessPoolExecutor(max_workers=workers) as ex:
   for a,b,c in ex.map(process_audio_group,groups,chunksize=1):rr.extend(a);ff.extend(b);ca.extend(c)
  rows=pd.DataFrame(rr,columns=OUT_COLS);fails=pd.DataFrame(ff,columns=['token_id','feature','failure_reason'])
  tmp=op.with_suffix('.parquet.tmp');rows.to_parquet(tmp,index=False,compression='zstd');os.replace(tmp,op);atomic_tsv(fails,fp);atomic_tsv(pd.DataFrame(ca),ap)
  m=dict(chunk=no,input_tokens=len(tok),flatness_tokens=rows.loc[rows.feature=='flatness','token_id'].nunique(),cpps_frozen_eligible=sum(tok.token_id.isin(cpp_ok)),cpps_complete_tokens=rows.loc[rows.feature=='cpps','token_id'].nunique(),rows=len(rows),failures=len(fails),elapsed_seconds=time.perf_counter()-t);done.write_text(json.dumps(m));man.append(m);print(json.dumps(m),flush=True)
  del x,q,tok,rows
 final=out/'extracted/flatness_cpp_trajectories.parquet';tmp=final.with_suffix('.parquet.tmp');writer=None
 for p in sorted(chunkdir.glob('extension_*.parquet')):
  pf=pq.ParquetFile(p)
  for batch in pf.iter_batches(batch_size=250000):
   tab=pa.Table.from_batches([batch]);writer=pq.ParquetWriter(tmp,tab.schema,compression='zstd') if writer is None else writer;writer.write_table(tab)
 writer.close();os.replace(tmp,final)
 # Feature-token eligibility and loss summaries.
 base=pd.read_csv(root/'rebuild_v6_trajectory/extracted/trajectory_eligible_tokens.tsv',sep='\t',low_memory=False)
 success=[];pf=pq.ParquetFile(final)
 for batch in pf.iter_batches(batch_size=500000,columns=['token_id','feature']): success.append(batch.to_pandas().drop_duplicates())
 suc=pd.concat(success).drop_duplicates();fparts=[pd.read_csv(p,sep='\t') for p in sorted(chunkdir.glob('extension_*_failures.tsv'))];fail=pd.concat(fparts,ignore_index=True) if fparts else pd.DataFrame(columns=['token_id','feature','failure_reason'])
 details=[]
 for feature in ('flatness','cpps'):
  z=base[['token_id','file_id','partition','system','label','speaker_id','phone_A','phone_B','phone_pair','boundary_class','boundary_time','A_duration_ms','B_duration_ms','audio_path','sample_rate','audio_frames']].copy();ok=set(suc.loc[suc.feature==feature,'token_id']);z['feature']=feature;z['eligibility']=z.token_id.isin(ok)
  reason=fail[fail.feature==feature].drop_duplicates('token_id').set_index('token_id').failure_reason.to_dict()
  if feature=='cpps': z['failure_reason']=z.token_id.map(lambda t:'' if t in ok else reason.get(t,'frozen_v5_cpps_ineligible_or_failed'))
  else:z['failure_reason']=z.token_id.map(lambda t:'' if t in ok else reason.get(t,'flatness_extraction_failed'))
  details.append(z)
 det=pd.concat(details,ignore_index=True);det.to_parquet(out/'extracted/feature_token_eligibility.parquet',index=False,compression='zstd')
 loss=det.groupby(['partition','system','feature','eligibility','failure_reason'],dropna=False).size().rename('tokens').reset_index();atomic_tsv(loss,out/'audit/token_loss_by_system_feature_reason.tsv')
 atomic_tsv(pd.DataFrame(man),out/'audit/extraction_chunk_summary.tsv')
 env=pd.DataFrame([('python',platform.python_version()),('numpy',np.__version__),('pandas',pd.__version__),('pyarrow',pa.__version__),('soundfile',sf.__version__),('parselmouth',parselmouth.__version__),('praat',parselmouth.PRAAT_VERSION)],columns=['component','version']);atomic_tsv(env,out/'audit/runtime_versions_extraction.tsv')
 print(json.dumps(dict(chunks=len(man),rows=sum(m['rows'] for m in man),flatness_tokens=int(det.query("feature=='flatness' and eligibility").token_id.nunique()),cpps_tokens=int(det.query("feature=='cpps' and eligibility").token_id.nunique()),output=str(final)),indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,required=True);ap.add_argument('--mode',choices=['audit','full'],required=True);ap.add_argument('--workers',type=int,default=max(1,min(12,(os.cpu_count() or 2)-1)));a=ap.parse_args();root=a.project_root.resolve();out=root/'rebuild_v6_trajectory/cpp_flatness_extension'
 for d in [out/'audit',out/'logs',out/'extracted']:d.mkdir(parents=True,exist_ok=True)
 params(out)
 if a.mode=='audit':audit_mode(root,out)
 else:full_mode(root,out,a.workers)
if __name__=='__main__':main()
