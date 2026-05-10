from __future__ import annotations

from search_r1.query_token_decomposition import (
    allocate_reward_to_tokens,
    build_query_token_masks,
    extract_search_spans,
)


class MockTokenizer:
    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        input_ids = []
        offsets = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            j = i + 1
            while j < len(text) and not text[j].isspace():
                j += 1
            offsets.append((i, j))
            input_ids.append(len(input_ids) + 1)
            i = j
        return {"input_ids": input_ids, "offset_mapping": offsets}

    def convert_ids_to_tokens(self, input_ids):
        return [f"tok_{i}" for i in input_ids]


def test_single_search():
    text = "<think><search>Deacon Blue members</search></think>"
    tk = MockTokenizer()
    spans = extract_search_spans(text)
    assert len(spans) == 1
    assert spans[0]["query_text"] == "Deacon Blue members"

    masks = build_query_token_masks(text, tk)
    q_idx = [i for i, v in enumerate(masks["query_token_mask"]) if v]
    a_idx = [i for i, v in enumerate(masks["action_mask"]) if v]
    assert len(q_idx) > 0
    assert set(q_idx).issubset(set(a_idx))

    scores, warn = allocate_reward_to_tokens(len(masks["tokens"]), q_idx, 1.0)
    assert warn is None
    assert abs(sum(scores) - 1.0) < 1e-7


def test_multiple_searches():
    text = "<search>alpha</search> and <search>beta gamma</search>"
    spans = extract_search_spans(text)
    assert len(spans) == 2


def test_invalid_search_format():
    text = '<search query="abc">'
    spans = extract_search_spans(text)
    assert len(spans) == 0


def test_empty_query_warning():
    text = "<search></search>"
    tk = MockTokenizer()
    masks = build_query_token_masks(text, tk)
    q_idx = [i for i, v in enumerate(masks["query_token_mask"]) if v]
    scores, warn = allocate_reward_to_tokens(len(masks["tokens"]), q_idx, 1.0)
    assert abs(sum(scores) - 0.0) < 1e-7
    assert warn is not None


if __name__ == "__main__":
    test_single_search()
    test_multiple_searches()
    test_invalid_search_format()
    test_empty_query_warning()
    print("all tests passed")
