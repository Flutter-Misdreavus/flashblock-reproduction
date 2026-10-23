#!/usr/bin/env python3
"""Read-only audit of frozen scorer, raw generations, scores and optional source data."""
import argparse,hashlib,json,random,subprocess,sys
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'repro'))
import quality_scorer as scorer
p=argparse.ArgumentParser()
for name in ['config','input-manifest','rows','raw-manifest','raw-root','out']:p.add_argument('--'+name,required=True)
p.add_argument('--tokenizer');p.add_argument('--dataset');a=p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
cfg=read(a.config);man=read(a.input_manifest);rows=read(a.rows)['rows'];rm=read(a.raw_manifest)
scorer_sha=sha(ROOT/'repro/quality_scorer.py');assert scorer_sha=='7f9225c8aef0018ace2d4f81f2f6bbfb4c2be10305c11b9afd21d58b47e21f8a'
ids=sorted(random.Random(42).sample(range(1319),128));assert man['dataset']['ids']==ids
assert cfg['controls']['ignore_eos'] is False and cfg['controls']['max_new_tokens']==4096
assert len(rows)==256 and len(rm['groups'])==64
by={(r['index'],r['condition']):r for r in rows};assert len(by)==256
meta={r['index']:r for r in man['items']};tok=None
if a.tokenizer:
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.tokenizer,use_fast=True,trust_remote_code=True)
gold_source_verified=0;prompts_verified=0
if a.dataset:
 import pyarrow as pa
 import pyarrow.parquet as pq
 pa.set_cpu_count(8);assert sha(a.dataset)==cfg['dataset']['test_sha256']
 data=pq.read_table(a.dataset).to_pylist();assert len(data)==1319
 for idx in ids:
  gold=Fraction(data[idx]['answer'].rsplit('####',1)[1].strip().replace(',',''))
  assert gold==Fraction(meta[idx]['gold_numeric']);gold_source_verified+=1
  if tok is not None:
   content=cfg['prompt']['instruction']+'\n'+data[idx]['question']
   prompt=tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
   assert prompt==cfg['prompt']['template'].replace('{problem}',data[idx]['question'])
   assert hashlib.sha256(prompt.encode()).hexdigest()==meta[idx]['prompt_sha256']
   assert len(tok.encode(prompt))==meta[idx]['prompt_tokens'];prompts_verified+=1
seen=set();group_hashes=[];decoded=0;rescored=0;last_tokens={};lengths=[]
for entry in rm['groups']:
 name=Path(entry['relpath']).name;matches=list(Path(a.raw_root).rglob(name));assert len(matches)==1,(name,matches)
 path=matches[0];assert sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes']
 d=read(path);g=d['group_index'];c=d['condition'];order=['dense','s2'] if g%2==0 else ['s2','dense']
 assert d['status']=='SUCCEEDED' and not d['problems'] and d['seed']==42 and d['max_new_tokens']==4096
 assert d['execution_commit']=='0b39fb9253818ac30e651230564e24bc89187bc1'
 assert d['group_condition_order']==order.index(c)
 assert d['config_sha256']==sha(a.config) and d['manifest_sha256']==sha(a.input_manifest) and d['scorer_sha256']==scorer_sha
 assert [r['index'] for r in d['items']]==ids[g*4:(g+1)*4]
 for raw in d['items']:
  key=(raw['index'],c);assert key not in seen;seen.add(key);r=by[key];tokens=raw['token_ids'];text=raw['text']
  assert len(tokens)==raw['new_tokens']==r['new_tokens'] and 0<len(tokens)<=4096
  assert hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()==raw['output_sha256']==r['output_sha256']
  assert raw['prompt_sha256']==r['prompt_sha256']==meta[key[0]]['prompt_sha256']
  assert r['gold_numeric']==meta[key[0]]['gold_numeric']
  assert r['execution_commit']==d['execution_commit'] and r['group_index']==g and r['group_condition_order']==order.index(c)
  pred=scorer.extract_prediction(text);canon=scorer.canonical_fraction(pred['value'])
  assert canon==r['prediction_numeric'] and pred['method']==r['extraction_method']
  assert r['correct']==(pred['value'] is not None and pred['value']==Fraction(meta[key[0]]['gold_numeric']))
  if tok is not None:
   assert all(isinstance(x,int) and 0<=x<len(tok) for x in tokens)
   assert tok.mask_token_id is not None and tok.mask_token_id not in tokens and tok.decode(tokens,skip_special_tokens=False)==text
   assert r['empty_output']==(not tok.decode(tokens,skip_special_tokens=True).strip())
   stop='eos' if tokens[-1]==tok.eos_token_id else ('max_tokens' if len(tokens)==4096 else 'other')
   assert stop==raw['finish_reason']==r['stop_reason'];assert r['truncated']==(stop=='max_tokens');decoded+=1
  last_tokens[str(tokens[-1])]=last_tokens.get(str(tokens[-1]),0)+1;lengths.append(len(tokens));rescored+=1
 group_hashes.append({'file':name,'sha256':sha(path)})
assert seen==set(by)
out={'status':'PASSED','script_sha256':sha(__file__),'execution_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'frozen_scorer_sha256':scorer_sha,'input_hashes':{k:sha(getattr(a,k)) for k in ['config','input_manifest','rows','raw_manifest']},'groups_verified':len(group_hashes),'raw_outputs_rehashed_and_rescored':rescored,'tokenizer_decodes_verified':decoded,'gold_answers_independently_verified_from_parquet':gold_source_verified,'prompts_reconstructed_and_tokenized':prompts_verified,'last_token_counts':last_tokens,'output_length_range':[min(lengths),max(lengths)],'group_hashes':group_hashes,'note':'No scorer or answer changed; pure replay of frozen scorer on raw text, exact rational comparison; optional independent tokenizer decode and raw dataset gold extraction.'}
Path(a.out).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['group_hashes','input_hashes']}))
