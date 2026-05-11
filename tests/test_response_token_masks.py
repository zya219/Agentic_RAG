import os,sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from search_r1.query_token_decomposition import build_response_token_masks
from search_r1.agentic_rag_reward import build_token_level_scores, build_token_level_scores_with_debug
import torch

class MockTokenizer:
    def __call__(self,text,return_offsets_mapping=True,add_special_tokens=False):
        ids=list(range(len(text))); offs=[(i,i+1) for i in range(len(text))]
        return {'input_ids':ids,'offset_mapping':offs}
    def convert_ids_to_tokens(self,ids): return [str(i) for i in ids]
    def decode(self,ids): return ''.join(chr(int(i)) for i in ids.tolist())

def test_valid_search_answer():
    t='<search>Deacon Blue members</search><answer>Deacon Blue</answer>'
    m=build_response_token_masks(t,MockTokenizer())
    assert sum(m['search_query_mask'])>0 and sum(m['answer_content_mask'])>0 and sum(m['format_mask'])>0
    assert m['invalid_search_count']==0

def test_invalid_search():
    m=build_response_token_masks('<search query="abc">',MockTokenizer())
    assert m['invalid_search_count']>0

def test_strict_no_fallback():
    text='</search>'
    ids=torch.tensor([[ord(c) for c in text]],dtype=torch.long)
    class Item: pass
    class B:
        def __init__(self):
            self.batch={'responses':ids,'attention_mask':torch.ones_like(ids),'prompts':torch.zeros((1,0),dtype=torch.long)}
            self.non_tensor_batch={}
            self.meta_info={'tokenizer':MockTokenizer()}
        def __len__(self):
            return self.batch['responses'].shape[0]
        def __getitem__(self,i):
            item=Item()
            item.batch={
                'responses':self.batch['responses'][i],
                'attention_mask':self.batch['attention_mask'][i],
                'prompts':torch.zeros((0,),dtype=torch.long),
            }
            item.non_tensor_batch={}
            return item
    b=B()
    s=build_token_level_scores(b,tokenizer=MockTokenizer(),reward_decomposition_mode='strict_query_token')
    assert float(s.sum().item())<=0.0

def test_cost_format_invalid_format_debug_cases():
    def run_case(text):
        ids=torch.tensor([[ord(c) for c in text]],dtype=torch.long)
        class Item: pass
        class B:
            def __init__(self):
                self.batch={'responses':ids,'attention_mask':torch.ones_like(ids),'prompts':torch.zeros((1,0),dtype=torch.long)}
                self.non_tensor_batch={}
                self.meta_info={'tokenizer':MockTokenizer()}
            def __len__(self): return self.batch['responses'].shape[0]
            def __getitem__(self,i):
                item=Item()
                item.batch={'responses':self.batch['responses'][i],'attention_mask':self.batch['attention_mask'][i],'prompts':torch.zeros((0,),dtype=torch.long)}
                item.non_tensor_batch={}
                return item
        _, dbg = build_token_level_scores_with_debug(B(), tokenizer=MockTokenizer(), reward_decomposition_mode='strict_query_token_cost_format')
        return dbg[0]

    d1 = run_case("<search>abc")
    assert d1["search_mismatch_count"] == 1
    assert d1["invalid_format_count"] == 1

    d2 = run_case("<search>a</search>")
    assert d2["answer_missing_count"] == 1
    assert d2["invalid_format_count"] == 1

    d3 = run_case("<search>a</search><answer>ok</answer>")
    assert d3["full_format_valid"] == 1
    assert d3["invalid_format_count"] == 0

if __name__=='__main__':
    test_valid_search_answer(); test_invalid_search(); test_strict_no_fallback(); test_cost_format_invalid_format_debug_cases(); print('ok')
