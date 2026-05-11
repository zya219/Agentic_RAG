#!/usr/bin/env python3
import argparse,csv,json,collections,os

def ensure_parent(path):
 parent=os.path.dirname(path)
 if parent:
  os.makedirs(parent, exist_ok=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input_jsonl',required=True);ap.add_argument('--output_markdown',default='results/token_reward_debug_summary.md');ap.add_argument('--output_csv',default='results/token_reward_debug_summary.csv');ap.add_argument('--output_json',default='results/token_reward_debug_summary.json');a=ap.parse_args()
 rows=[json.loads(x) for x in open(a.input_jsonl) if x.strip()]
 n=len(rows)
 def avg(v): return sum(v)/n if n else 0.0
 wc=collections.Counter()
 for r in rows:
  for w in r.get('warnings',[]): wc[w]+=1
 s={
 'total_cases':n,
 'cases_with_search':sum(1 for r in rows if any('QUERY' in m for m in r.get('masks',[]))),
 'cases_with_answer':sum(1 for r in rows if any('ANSWER' in m for m in r.get('masks',[]))),
 'avg_query_token_count':avg([sum(1 for m in r.get('masks',[]) if 'QUERY' in m) for r in rows]),
 'avg_answer_content_token_count':avg([sum(1 for m in r.get('masks',[]) if 'ANSWER' in m) for r in rows]),
 'avg_format_token_count':avg([sum(1 for m in r.get('masks',[]) if 'FORMAT' in m) for r in rows]),
 'avg_invalid_search_count':avg([r.get('invalid_search_count',0) for r in rows]),
 'avg_total_reward_sum':avg([r.get('reward_sums',{}).get('total',0.0) for r in rows]),
 'total_token_reward_sum':sum(r.get('reward_sums',{}).get('total',0.0) for r in rows),
 'positive_token_reward_sum':sum(sum(max(0.0,v) for v in r.get('token_rewards',[])) for r in rows),
 'negative_token_reward_sum':sum(sum(min(0.0,v) for v in r.get('token_rewards',[])) for r in rows),
 'nonzero_reward_token_count':sum(sum(1 for v in r.get('token_rewards',[]) if abs(v)>1e-12) for r in rows),
 'query_token_reward_sum':sum(sum(v for v,m in zip(r.get('token_rewards',[]),r.get('masks',[])) if 'QUERY' in m) for r in rows),
 'answer_token_reward_sum':sum(sum(v for v,m in zip(r.get('token_rewards',[]),r.get('masks',[])) if 'ANSWER' in m) for r in rows),
 'format_token_reward_sum':sum(sum(v for v,m in zip(r.get('token_rewards',[]),r.get('masks',[])) if 'FORMAT' in m) for r in rows),
 'search_count':sum(r.get('search_count',0) for r in rows),
 'repeated_search_count':sum(r.get('repeated_search_count',0) for r in rows),
 'invalid_format_count':sum(r.get('invalid_format_count',0) for r in rows),
 'answer_missing_count':sum(r.get('answer_missing_count',0) for r in rows),

 'search_mismatch_count':sum(r.get('search_mismatch_count',0) for r in rows),
 'answer_mismatch_count':sum(r.get('answer_mismatch_count',0) for r in rows),
 'malformed_action_count':sum(r.get('malformed_action_count',0) for r in rows),
 'full_format_valid_count':sum(r.get('full_format_valid',0) for r in rows),
 'avg_search_mismatch_count':avg([r.get('search_mismatch_count',0) for r in rows]),
 'avg_answer_mismatch_count':avg([r.get('answer_mismatch_count',0) for r in rows]),
 'avg_malformed_action_count':avg([r.get('malformed_action_count',0) for r in rows]),
 'avg_full_format_valid':avg([r.get('full_format_valid',0) for r in rows]),
 'max_reward_conservation_error':max([abs(r.get('reward_sums',{}).get('conservation_error',0.0)) for r in rows], default=0.0),
 'warning_counts':dict(wc)}
 ensure_parent(a.output_markdown); ensure_parent(a.output_csv); ensure_parent(a.output_json)
 open(a.output_json,'w').write(json.dumps(s,ensure_ascii=False,indent=2))
 with open(a.output_csv,'w',newline='') as f:
  w=csv.writer(f); w.writerow(['metric','value']); [w.writerow([k,v]) for k,v in s.items() if k!='warning_counts']
 open(a.output_markdown,'w').write('# Token Reward Debug Summary\n\n'+''.join([f'- {k}: {v}\n' for k,v in s.items() if k!='warning_counts'])+'\n## warning_counts\n'+''.join([f'- {k}: {v}\n' for k,v in s['warning_counts'].items()]))
if __name__=='__main__': main()
