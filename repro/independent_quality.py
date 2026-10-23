#!/usr/bin/env python3
"""Independent paired quality aggregation; exact rational EM, item bootstrap."""
import argparse, collections, hashlib, json, random, subprocess
from fractions import Fraction
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args()
src=Path(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
payload=json.loads(src.read_text());rows=payload['rows'];assert len(rows)==256
ids=sorted(random.Random(42).sample(range(1319),128));by={}
for r in rows:
 key=(r['index'],r['condition']);assert key not in by;by[key]=r
 assert r['index'] in ids and r['condition'] in ['dense','s2']
 g=ids.index(r['index'])//4;assert r['group_index']==g
 expected_order=['dense','s2'] if g%2==0 else ['s2','dense'];assert r['group_condition_order']==expected_order.index(r['condition'])
 assert r['ids_in_vocab'] and not r['contains_mask_token']
 assert 0<=r['new_tokens']<=4096
 gold=Fraction(r['gold_numeric']);pred=None if r['prediction_numeric'] is None else Fraction(r['prediction_numeric'])
 assert r['correct']==(pred is not None and pred==gold)
 assert not r['truncated'] or r['new_tokens']==4096
for i in ids:assert by[(i,'dense')]['gold_numeric']==by[(i,'s2')]['gold_numeric'] and by[(i,'dense')]['prompt_sha256']==by[(i,'s2')]['prompt_sha256']
dense=np.array([by[(i,'dense')]['correct'] for i in ids],dtype=int);s2=np.array([by[(i,'s2')]['correct'] for i in ids],dtype=int);delta=s2-dense
rng=np.random.default_rng(20261023);boot=delta[rng.integers(0,128,size=(10000,128))].mean(axis=1)*100
# Exact bootstrap distribution of sum of iid draws from {-1,0,1} empirical delta.
prob=np.array([(delta==-1).mean(),(delta==0).mean(),(delta==1).mean()]);pmf=np.array([1.])
for _ in range(128):pmf=np.convolve(pmf,prob)
cdf=pmf.cumsum();support=np.arange(-128,129)/128*100
exact=[float(support[np.searchsorted(cdf,q)]) for q in [.025,.975]]
disc={'both_correct':int(((dense==1)&(s2==1)).sum()),'dense_only':int(((dense==1)&(s2==0)).sum()),'s2_only':int(((dense==0)&(s2==1)).sum()),'neither':int(((dense==0)&(s2==0)).sum())}
conds={}
for c,correct in [('dense',dense),('s2',s2)]:
 rr=[by[(i,c)] for i in ids];lens=np.array([r['new_tokens'] for r in rr]);methods=collections.Counter(r['extraction_method'] for r in rr)
 conds[c]={'correct':int(correct.sum()),'n':128,'em_percent':float(correct.mean()*100),'new_tokens_mean':float(lens.mean()),'new_tokens_median':float(np.median(lens)),'new_tokens_p95':float(np.quantile(lens,.95)),'new_tokens_max':int(lens.max()),'truncated':sum(r['truncated'] for r in rr),'empty_outputs':sum(r['empty_output'] for r in rr),'extraction_failures':sum(r['prediction_numeric'] is None for r in rr),'extraction_method_counts':dict(methods),'by_method':{m:{'n':sum(r['extraction_method']==m for r in rr),'correct':sum(r['correct'] for r in rr if r['extraction_method']==m)} for m in methods},'stop_reason_counts':dict(collections.Counter(r['stop_reason'] for r in rr))}
result={'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'analysis_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'run_commits':sorted(set(r['execution_commit'] for r in rows)),'n':128,'conditions':conds,'paired_delta_pp':float(delta.mean()*100),'paired_bootstrap_ci95_pp':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_seed':20261023,'bootstrap_repetitions':10000,'exact_paired_bootstrap_ci95_pp':exact,'exact_bootstrap_method':'128-fold convolution of empirical paired delta {-1,0,+1}; inverse CDF quantiles; no normal approximation','discordance':disc,'more_than_5pp_loss':bool(delta.mean()*100 < -5),'limitation':'fixed128 items, one seed, original undistilled model; no equivalence or paper-table claim'}
(out/'independent-review.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(9,3.8));vals=[conds[c]['em_percent'] for c in ['dense','s2']]
bars=ax[0].bar(['dense','s2'],vals,color=['#465b75','#db6b35']);ax[0].set_ylim(0,100);ax[0].set_ylabel('Numeric exact match (%)')
for rect,c in zip(bars,['dense','s2']):ax[0].text(rect.get_x()+rect.get_width()/2,rect.get_height()+2,f"{conds[c]['correct']}/128",ha='center')
m=result['paired_delta_pp'];lo,hi=exact;ax[1].errorbar(m,0,xerr=[[m-lo],[hi-m]],fmt='o',capsize=5,color='#247c83');ax[1].axvline(0,color='gray',ls='--');ax[1].axvline(-5,color='#b84d4d',ls=':');ax[1].set_yticks([0],['s2 − dense']);ax[1].set_xlabel('Paired EM difference (percentage points)');ax[1].set_title(f'{m:+.2f} pp; exact bootstrap 95% CI')
fig.suptitle('GSM8K fixed128 | original TraDo4B BF16 | single seed42');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'quality-paired.{ext}',dpi=180)
print(json.dumps(result))
