import os,sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from search_r1.query_token_decomposition import build_response_token_masks
from search_r1.agentic_rag_reward import build_token_level_scores
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
    class B: pass
    b=B(); b.batch={'responses':ids,'attention_mask':torch.ones_like(ids),'prompts':torch.zeros((1,0),dtype=torch.long)}; b.non_tensor_batch={}; b.meta_info={'tokenizer':MockTokenizer()}
    def __len__(self): return 1
    def __getitem__(self,i): return self
    B.__len__=__len__; B.__getitem__=__getitem__
    s=build_token_level_scores(b,tokenizer=MockTokenizer(),reward_decomposition_mode='strict_query_token')
    assert float(s.sum().item())<=0.0

if __name__=='__main__':
    test_valid_search_answer(); test_invalid_search(); test_strict_no_fallback(); print('ok')
