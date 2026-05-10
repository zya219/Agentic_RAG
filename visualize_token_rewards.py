#!/usr/bin/env python3
import argparse, json
from transformers import AutoTokenizer
from search_r1.query_token_decomposition import build_response_token_masks
from search_r1.agentic_rag_reward import compute_answer_correctness_reward

def allocate(n, idxs, r):
    s=[0.0]*n
    idxs=sorted(set(i for i in idxs if 0<=i<n))
    if not idxs:return s
    v=r/len(idxs)
    for i in idxs:s[i]=v
    return s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input_jsonl',required=True);ap.add_argument('--output_markdown',required=True);ap.add_argument('--output_jsonl',required=True)
    ap.add_argument('--model_id',default='Qwen/Qwen2.5-3B-Instruct');ap.add_argument('--max_samples',type=int,default=5)
    ap.add_argument('--mode',choices=['none','coarse_action','query_token_uniform','strict_query_token'],default='none')
    ap.add_argument('--text_field',default='result');ap.add_argument('--search_reward_value',type=float,default=1.0)
    ap.add_argument('--answer_reward_value',type=float,default=1.0);ap.add_argument('--format_reward_value',type=float,default=0.0)
    ap.add_argument('--format_penalty_value',type=float,default=-1.0);ap.add_argument('--require_search',action='store_true');ap.add_argument('--require_answer',action='store_true')
    a=ap.parse_args(); tok=AutoTokenizer.from_pretrained(a.model_id)
    rows=[]
    with open(a.input_jsonl) as f:
        for ln in f:
            if ln.strip(): rows.append(json.loads(ln))
    out=[]; md=['# Token Reward Visualization\n']
    for r in rows:
        txt=r.get(a.text_field,'') or ''
        m=build_response_token_masks(txt,tok)
        if a.require_search and not m['search_actions']: continue
        if a.require_answer and not m['answer_actions']: continue
        n=len(m['tokens']); rewards=[0.0]*n
        if a.mode=='coarse_action':
            rewards=[x+y for x,y in zip(rewards,allocate(n,[i for i,v in enumerate(m['action_mask']) if v],a.search_reward_value))]
        elif a.mode in ('query_token_uniform','strict_query_token'):
            sr=0.0 if (a.mode=='strict_query_token' and m['invalid_search_count']>0) else a.search_reward_value
            rewards=[x+y for x,y in zip(rewards,allocate(n,[i for i,v in enumerate(m['search_query_mask']) if v],sr))]
            if a.mode=='strict_query_token':
                rewards=[x+y for x,y in zip(rewards,allocate(n,[i for i,v in enumerate(m['answer_content_mask']) if v],a.answer_reward_value))]
                fmt=[i for i,v in enumerate(m['format_mask']) if v]
                fr=a.format_reward_value if m['invalid_search_count']==0 else a.format_penalty_value
                if not fmt and m['invalid_search_count']>0 and n>0: fmt=[n-1]
                rewards=[x+y for x,y in zip(rewards,allocate(n,fmt,fr))]
        masks=[]
        for i in range(n):
            labs=[]
            if m['format_mask'][i]: labs.append('FORMAT')
            if m['search_query_mask'][i]: labs.append('QUERY')
            if m['answer_content_mask'][i]: labs.append('ANSWER')
            if m['action_mask'][i]: labs.append('ACTION')
            masks.append(labs)
        item={'id':r.get('id'),'question':r.get('question'),'tokens':m['tokens'],'offsets':m['offsets'],'masks':masks,'token_rewards':rewards,'reward_sums':{'total':sum(rewards)},'invalid_search_count':m['invalid_search_count'],'warnings':m['warnings']}
        out.append(item)
        md.append(f"## id={item['id']}\n")
        md.append(f"invalid_search_count: {item['invalid_search_count']}\n\n")
        md.append('|index|token|offset|masks|token_reward|\n|---:|---|---|---|---:|\n')
        for i,t in enumerate(item['tokens']): md.append(f"|{i}|{str(t).replace('|','\\|')}|{item['offsets'][i]}|{','.join(masks[i])}|{rewards[i]:.6f}|\n")
        if len(out)>=a.max_samples: break
    open(a.output_markdown,'w').write(''.join(md))
    with open(a.output_jsonl,'w') as f:
        for o in out: f.write(json.dumps(o,ensure_ascii=False)+'\n')
if __name__=='__main__': main()
