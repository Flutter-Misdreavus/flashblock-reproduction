#!/usr/bin/env python3
"""Independent paired-timing audit; exhaustive 5^5 paired bootstrap, no GPU."""
import argparse, hashlib, itertools, json, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--profile');a=p.parse_args()
src=Path(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
payload=json.loads(src.read_text());rows=payload['rows'];assert len(rows)==60,len(rows)
settings=[(b,l) for b in [1,4] for l in [1024,4096,8192]]
expected=[(b,l,r,c) for b,l in settings for r in range(1,6) for c in (['dense','s2'] if r%2 else ['s2','dense'])]
actual=[(x['batch'],x['length'],x['round'],x['condition']) for x in rows]
assert actual==expected,'missing, duplicate or out-of-order timing rows'
for r in rows:
 assert r['output_token_counts']==[256]*r['batch']
 assert r['total_output_tokens']==r['batch']*256
 assert not r['problems']
 assert r['seed']==42 and r['ignore_eos']
 assert np.isfinite(r['decode_seconds']) and r['decode_seconds']>0
 assert np.isclose(r['decode_tps'],r['total_output_tokens']/r['decode_seconds'])
 assert np.isclose(r['e2e_seconds'],r['prefill_seconds']+r['decode_seconds'])
indices=np.array(list(itertools.product(range(5),repeat=5)))
results={}
for b,l in settings:
 s=[r for r in rows if r['batch']==b and r['length']==l]
 conds={c:sorted([r for r in s if r['condition']==c],key=lambda r:r['round']) for c in ['dense','s2']}
 times={c:np.array([r['decode_seconds'] for r in v]) for c,v in conds.items()}
 ratios=times['dense']/times['s2'];boot=ratios[indices].mean(axis=1)
 stats={'batch':b,'initial_tokens_per_sequence':l,'ratios':ratios.tolist(),'mean_paired_speedup':float(ratios.mean()),'median_paired_speedup':float(np.median(ratios)),'exhaustive_paired_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'conditions':{}}
 for c,rr in conds.items():
  t=times[c];tp=np.array([r['decode_tps'] for r in rr]);ctr={k:sum(r['counters'].get(k,0) for r in rr) for k in ['decode_steps','decode_rows','reuse_rows','update_rows','store_rows','saving_rows']}
  stats['conditions'][c]={'decode_seconds':t.tolist(),'decode_seconds_mean':float(t.mean()),'decode_tps_mean':float(tp.mean()),'decode_tps_median':float(np.median(tp)),'decode_tps_aggregate':float(b*256*5/t.sum()),'prefill_seconds_mean':float(np.mean([r['prefill_seconds'] for r in rr])),'e2e_seconds_mean':float(np.mean([r['e2e_seconds'] for r in rr])),'counters':ctr,'reuse_fraction_all_decode_rows':ctr['reuse_rows']/ctr['decode_rows'] if ctr['decode_rows'] else None,'reuse_fraction_denoising_rows':ctr['reuse_rows']/(ctr['decode_rows']-ctr['saving_rows']) if ctr['decode_rows']>ctr['saving_rows'] else None,'denoising_rows_per_run_mean':(ctr['decode_rows']-ctr['saving_rows'])/5,'decode_calls_per_run_mean':ctr['decode_steps']/5,'distinct_output_hashes':len(set(r['output_concat_sha256'] for r in rr)),'peak_allocated_mib_max':max(r['peak_allocated_mib'] for r in rr),'physical_free_boundary_min_mib':min(min(r['load_start']['mem_free_mib'],r['load_end']['mem_free_mib']) for r in rr),'decode_cv':float(t.std(ddof=1)/t.mean())}
 results[f'B{b}L{l}']=stats
result={'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'analysis_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'run_commits':sorted(set(r['execution_commit'] for r in rows)),'validation':'60 rows; fixed order, counts, seed, termination and arithmetic pass','context_curve_caveat':'Different frozen training-text workloads are used at each context length; length-curve differences are descriptive and not isolated causal effects of length.','bootstrap':'exhaustive paired bootstrap: all 3125 length-5 resamples, percentile CI with linear interpolation; descriptive only on shared GPU','settings':results,'primary':results['B4L8192']}
# Correct CPU-only attribution: flashinfer::sampling is sampling, not attention.
if a.profile:
 profile_src=Path(a.profile);pr=json.loads(profile_src.read_text());corrected={}
 for run in pr['runs']:
  kernels=run['profile']['kernels'];groups={'attention_custom_flash':0.,'flashinfer_sampling':0.,'other_device_work':0.};classified=[]
  for k in kernels:
   name=k['key'];category='flashinfer_sampling' if 'flashinfer::sampling::' in name else ('attention_custom_flash' if name.startswith('void flash::') else 'other_device_work')
   groups[category]+=k['self_cuda_us']/1e6;classified.append({'key':name,'category':category,'seconds':k['self_cuda_us']/1e6,'count':k['count']})
  assert np.isclose(sum(groups.values()),run['profile']['totals']['total_cuda_us']/1e6)
  corrected[run['condition']]={'seconds':groups,'total_device_seconds':sum(groups.values()),'attention_fraction':groups['attention_custom_flash']/sum(groups.values()),'legacy_worker_attention_seconds':run['profile']['totals']['attention_cuda_us']/1e6,'kernels':classified}
 result['profile_review']={'input_sha256':hashlib.sha256(profile_src.read_bytes()).hexdigest(),'rules':'Attention: exact void flash:: namespace (audited complete kernel list); flashinfer::sampling:: separately attributed to sampling; all others other device work. CPU review only; original raw profiler output unchanged. Worker broad flash pattern incorrectly includes FlashInfer sampling.','scope':'One separate instrumented profile per condition; device durations are not end-to-end walltimes; unknown native operations remain other, no intrinsic-kernel speedup inference.','conditions':corrected}
 figp,axp=plt.subplots(figsize=(7,4));left=np.zeros(2)
 for key,label,color in [('attention_custom_flash','Custom attention','#247c83'),('flashinfer_sampling','FlashInfer sampling','#db6b35'),('other_device_work','Other device work','#465b75')]:
  vv=[corrected[c]['seconds'][key] for c in ['dense','s2']];axp.barh(['dense','s2'],vv,left=left,label=label,color=color);left+=vv
 axp.set_xlabel('Sum of profiled device durations (seconds)');axp.set_title('Separate B4 × 8,192 profiles | CPU-audited attribution');axp.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,fontsize=8);figp.tight_layout()
 for ext in ['png','pdf']:figp.savefig(out/f'efficiency-profile.{ext}',dpi=180,bbox_inches='tight')

(out/'independent-review.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex='col')
for col,b in enumerate([1,4]):
 ls=[1024,4096,8192]
 for c,color in [('dense','#465b75'),('s2','#db6b35')]:
  ys=[results[f'B{b}L{l}']['conditions'][c]['decode_tps_mean'] for l in ls]
  axes[0,col].plot(ls,ys,'o-',label=c,color=color)
 axes[0,col].set_title(f'Batch {b}');axes[0,col].set_ylabel('Decode tokens/s (mean of 5 runs)');axes[0,col].legend()
 for l in ls:
  v=results[f'B{b}L{l}'];m=v['mean_paired_speedup'];lo,hi=v['exhaustive_paired_bootstrap_ci95']
  axes[1,col].errorbar(l,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color='#247c83')
 axes[1,col].plot(ls,[results[f'B{b}L{l}']['mean_paired_speedup'] for l in ls],color='#247c83')
 axes[1,col].axhline(1,color='gray',ls='--',lw=1);axes[1,col].axhline(1.1,color='#b84d4d',ls=':',lw=1)
 axes[1,col].set_ylabel('Mean paired dense / s2 decode time');axes[1,col].set_xlabel('Initial tokens per sequence');axes[1,col].set_xticks(ls,['1,024','4,096','8,192'])
fig.suptitle('FlashBlock efficiency | original TraDo4B BF16 | shared RTX3090')
fig.text(.5,.005,'256 new tokens/sequence; 5 paired interleaved repetitions; descriptive exhaustive bootstrap 95% CI',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.025,1,.96))
for ext in ['png','pdf']:fig.savefig(out/f'efficiency-context.{ext}',dpi=180)
print(json.dumps({'primary':result['primary'],'output':str(out)}))
