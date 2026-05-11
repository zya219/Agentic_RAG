#!/usr/bin/env python3
import argparse, json, os
from search_r1.query_token_decomposition import build_response_token_masks

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
    ap.add_argument('--mode',choices=['none','coarse_action','query_token_uniform','strict_query_token','strict_query_token_cost'],default='none')
    ap.add_argument('--text_field',default='result');ap.add_argument('--search_reward_value',type=float,default=1.0)
    ap.add_argument('--answer_reward_value',type=float,default=1.0);ap.add_argument('--format_reward_value',type=float,default=0.0)
    ap.add_argument('--format_penalty_value',type=float,default=-1.0);ap.add_argument('--require_search',action='store_true');ap.add_argument('--require_answer',action='store_true')
    ap.add_argument('--search_cost_value',type=float,default=0.1);ap.add_argument('--repeat_search_penalty_value',type=float,default=0.2);ap.add_argument('--answer_missing_penalty_value',type=float,default=-1.0)
    a=ap.parse_args()
    from transformers import AutoTokenizer

    tok=AutoTokenizer.from_pretrained(a.model_id)
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
        elif a.mode in ('query_token_uniform','strict_query_token','strict_query_token_cost'):
            sr=0.0 if (a.mode=='strict_query_token' and m['invalid_search_count']>0) else a.search_reward_value
            rewards=[x+y for x,y in zip(rewards,allocate(n,[i for i,v in enumerate(m['search_query_mask']) if v],sr))]
            if a.mode in ('strict_query_token','strict_query_token_cost'):
                if a.mode=='strict_query_token_cost':
                    rewards=[0.0]*n
                    valid_search=[s for s in m.get('search_actions',[]) if s.get('query_token_indices') and (s.get('query_text') or '').strip()]
                    for si,s in enumerate(valid_search):
                        qidx=s['query_token_indices']
                        rewards=[x+y for x,y in zip(rewards,allocate(n,qidx,a.search_reward_value))]
                        rewards=[x+y for x,y in zip(rewards,allocate(n,qidx,-a.search_cost_value))]
                        if si>=1:
                            rewards=[x+y for x,y in zip(rewards,allocate(n,qidx,-a.repeat_search_penalty_value))]
                rewards=[x+y for x,y in zip(rewards,allocate(n,[i for i,v in enumerate(m['answer_content_mask']) if v],a.answer_reward_value))]
                fmt=[i for i,v in enumerate(m['format_mask']) if v]
                invalid_format = (m['invalid_search_count']>0) or (m.get('invalid_answer_count',0)>0) or (m.get('empty_search_count',0)>0)
                fr=a.format_reward_value if not invalid_format else a.format_penalty_value
                if not fmt and invalid_format and n>0: fmt=[n-1]
                rewards=[x+y for x,y in zip(rewards,allocate(n,fmt,fr))]
                has_valid_answer=any((x.get('answer_text') or '').strip() for x in m.get('answer_actions',[]))
                if a.mode=='strict_query_token_cost' and not has_valid_answer:
                    rewards=[x+y for x,y in zip(rewards,allocate(n,fmt or ([n-1] if n>0 else []),a.answer_missing_penalty_value))]
        masks=[]
        for i in range(n):
            labs=[]
            if m['format_mask'][i]: labs.append('FORMAT')
            if m['search_query_mask'][i]: labs.append('QUERY')
            if m['answer_content_mask'][i]: labs.append('ANSWER')
            if m['action_mask'][i]: labs.append('ACTION')
            masks.append(labs)
        item={'id':r.get('id'),'question':r.get('question'),'tokens':m['tokens'],'offsets':m['offsets'],'masks':masks,'token_rewards':rewards,'reward_sums':{'total':sum(rewards)},'invalid_search_count':m['invalid_search_count'],'invalid_format_count':int((m['invalid_search_count']>0) or (m.get('invalid_answer_count',0)>0) or (m.get('empty_search_count',0)>0)),'search_count':sum(1 for s in m.get('search_actions',[]) if (s.get('query_text') or '').strip()),'repeated_search_count':max(0,sum(1 for s in m.get('search_actions',[]) if (s.get('query_text') or '').strip())-1),'answer_missing_count':int(not any((x.get('answer_text') or '').strip() for x in m.get('answer_actions',[]))),'search_query_mask':m.get('search_query_mask',[]),'answer_content_mask':m.get('answer_content_mask',[]),'format_mask':m.get('format_mask',[]),'action_mask':m.get('action_mask',[]),'warnings':m['warnings']}
        out.append(item)
        md.append(f"## id={item['id']}\n")
        md.append(f"invalid_search_count: {item['invalid_search_count']}\n\n")
        md.append('|index|token|offset|masks|token_reward|\n|---:|---|---|---|---:|\n')
        for i,t in enumerate(item['tokens']):
            safe_token = str(t).replace('|', '\\|')
            mask_text = ','.join(masks[i])
            md.append(f"|{i}|{safe_token}|{item['offsets'][i]}|{mask_text}|{rewards[i]:.6f}|\n")
        if len(out)>=a.max_samples: break
    os.makedirs(os.path.dirname(a.output_markdown) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(a.output_jsonl) or '.', exist_ok=True)
    open(a.output_markdown,'w').write(''.join(md))
    with open(a.output_jsonl,'w') as f:
        for o in out: f.write(json.dumps(o,ensure_ascii=False)+'\n')
if __name__=='__main__': main()
