import argparse
import httpx
import requests
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import torch
import transformers
import asyncio
from transformers.generation.stopping_criteria import StoppingCriteriaList, StopStringCriteria
from typing import Optional
import os
import pandas as pd
from tqdm.asyncio import tqdm as tqdm_async

from prompt import AGENT_PROMPT_V2_SHORT, DIRECT_PROMPT
from utils import extract_str_between, load_jsonl, write_jsonl

import re

RETRIEVER_URL = "http://127.0.0.1:8000/retrieve"

def decide_retrieval(question: str) -> bool:
    """
    Decide whether the current question should go through retrieval path.
    Heuristic version based on question type rather than specific memorized examples.
    """
    q = question.lower().strip()
    q = re.sub(r"\s+", " ", q)

    # 1) 明显不需要检索：简单数学 / 明显短小直接问答
    if re.fullmatch(r"[\d\s\+\-\*/\(\)\.=]+", q):
        return False

    if re.match(r"^(what is|what's)\s+\d+(\s*[\+\-\*/]\s*\d+)+\??$", q):
        return False

    # 2) 明显需要检索：时效性 / 当前信息 / 外部事实依赖强
    time_sensitive_markers = [
        "current", "currently", "latest", "recent", "today", "now",
        "stock price", "weather", "temperature", "exchange rate",
        "president of", "prime minister of", "ceo of", "mayor of",
    ]
    if any(marker in q for marker in time_sensitive_markers):
        return True

    # 3) 明显需要检索：具体事实关系查询
    strong_fact_patterns = [
        r"^who wrote\b",
        r"^who invented\b",
        r"^who founded\b",
        r"^who discovered\b",
        r"^when was\b",
        r"^where was\b",
        r"\bborn\b",
        r"\bdied\b",
        r"\bwhat year\b",
        r"\bwhich year\b",
        r"\bwho is the president of\b",
        r"\bwho is the ceo of\b",
    ]
    if any(re.search(pattern, q) for pattern in strong_fact_patterns):
        return True

    # 4) 明显可以 direct：基础定义 / 解释类问题
    direct_reasoning_patterns = [
        r"^what is\b",
        r"^what are\b",
        r"^define\b",
        r"^explain\b",
        r"^why\b",
        r"^how\b",
    ]
    if any(re.search(pattern, q) for pattern in direct_reasoning_patterns):
        # 但如果同时带有强事实特征，前面已经 return True 了
        return False

    # 5) 首都类问题默认先 direct
    # 原因：很多国家首都属于高频常识，不必因为 "capital" 一词就强制检索
    if re.search(r"\bcapital of\b", q):
        return False

    # 6) 兜底策略：默认 direct
    return False

_HTTPX_CLIENT_ASYNC: Optional[httpx.AsyncClient] = None


# Helper functions
def extract_search_query(text):
    """Extract the last search query from the text."""
    matches = extract_str_between(text, "<search>", "</search>")
    return matches[-1].strip() if matches else None


def extract_final_answer(text):
    """Extract the final answer from the complete response."""
    matches = extract_str_between(text, "<answer>", "</answer>")
    return matches[-1].strip() if matches else None

def extract_search_queries(text: str) -> list[str]:
    matches = extract_str_between(text, "<search>", "</search>")
    return [m.strip() for m in matches if m.strip()]

def strip_to_assistant_answer(text: str) -> str:
    """
    Keep only the assistant response after the chat template prefix.
    """
    marker = "<|im_start|>assistant"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()

def search(query, retriever_url: Optional[str] = None):
    """Perform search using the search endpoint."""
    try:
        payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
        url = retriever_url or RETRIEVER_URL
        response = requests.post(url, json=payload)
        results = response.json()['result']
        
        # Format search results
        formatted_results = []
        for idx, doc_item in enumerate(results[0]):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            formatted_results.append(f"Doc {idx+1}(Title: {title}) {text}")
        
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {str(e)}"


async def get_httpx_client_async() -> httpx.AsyncClient:
    """Return a shared AsyncClient, creating it on first use."""
    global _HTTPX_CLIENT_ASYNC
    if _HTTPX_CLIENT_ASYNC is None:
        _HTTPX_CLIENT_ASYNC = httpx.AsyncClient()
    return _HTTPX_CLIENT_ASYNC


async def close_httpx_client_async() -> None:
    """Close and clear the shared AsyncClient if it exists."""
    global _HTTPX_CLIENT_ASYNC
    if _HTTPX_CLIENT_ASYNC is not None:
        await _HTTPX_CLIENT_ASYNC.aclose()
        _HTTPX_CLIENT_ASYNC = None


async def search_async(query: str, retriever_url: Optional[str] = None) -> str:
    """Perform search asynchronously using `httpx.AsyncClient`.

    If a `client` is provided, it will be reused. Otherwise a module-level
    shared client is created on first use and reused across calls.
    """
    try:
        payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True,
        }
        search_client = await get_httpx_client_async()
        url = retriever_url or RETRIEVER_URL
        response = await search_client.post(url, json=payload)

        response.raise_for_status()
        results = response.json()['result']

        formatted_results = []
        for idx, doc_item in enumerate(results[0]):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            formatted_results.append(f"Doc {idx+1}(Title: {title}) {text}")

        return "\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {str(e)}"


def inference_hf_single(question: str, model, tokenizer, prompt: str = AGENT_PROMPT_V2_SHORT, retriever_url: Optional[str] = None) -> str:
    """
    Performs inference using an agentic RAG LLM with the specified XML format.
    
    Args:
        question: The user's question to answer
        model_id: The Hugging Face model ID to use
        search_endpoint: The search API endpoint URL
    
    Returns:
        The final answer extracted from the model's response
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Format question
    question = question.strip()
    if question[-1] != '?':
        question += '?'
    
    # Model-specific configurations (adjust based on your model)
    curr_eos = [151645, 151643]  # For Qwen2.5 series models

    # Prepare the initial input (prompt and question only)
    input_text = f"{prompt}\nUser Question: {question}"
    think_step_reasoning = "\n<think>\n<step>\n    <reasoning>"

    # Stop string lists for different tags
    stop_on_search_list = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stop_on_answer_list = ["</answer>", " </answer>", "</answer>\n", " </answer>\n"]

    # Combine all stop strings for the main stopping criteria
    all_stop_strings = stop_on_search_list + stop_on_answer_list
    stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, all_stop_strings)])
    
    # Apply chat template if available
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": input_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        full_response = prompt + think_step_reasoning
    else:
        full_response = input_text + think_step_reasoning
    
    # Main inference loop
    step_count = 0
    max_steps = 10  # Prevent infinite loops
    
    while step_count < max_steps:
        # Encode current prompt - use tokenizer() method for cleaner encoding
        inputs = tokenizer(full_response, return_tensors='pt').to(device)
        
        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # Decode generated tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_response += output_text
        
        # Check if generation ended with </answer> or EOS token (complete response)
        if "</answer>" in output_text or outputs[0][-1].item() in curr_eos:
            break
        
        # Check if we need to perform a search
        search_query = extract_search_query(output_text)
        
        if search_query:  # Continue the generation with search results in context
            search_results = search(search_query, retriever_url=retriever_url)
            full_response += f"\n    <context>{search_results}</context>\n    <conclusion>"
        else:
            full_response += "\n    <conclusion>"
            
        step_count += 1
    
    # print(f"Full response: {full_response}")
    # return full_response
    clean_response = strip_to_assistant_answer(full_response)
    print(f"Full response: {clean_response}")
    return clean_response

def inference_hf_single_direct(
    question: str,
    model,
    tokenizer,
    direct_prompt: str,
    max_new_tokens: int = 512
) -> str:
    """
    Direct-answer path: answer the question directly without retrieval.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    question = question.strip()
    if question and question[-1] != '?':
        question += '?'

    input_text = direct_prompt.format(question=question)

    if tokenizer.chat_template:
        messages = [{"role": "user", "content": input_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        prompt_text = input_text

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    # generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    # output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # return output_text
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    clean_response = strip_to_assistant_answer(output_text)
    return clean_response

def inference_hf(
    question: list[str] | str,
    model_id: str,
    tokenizer_id: Optional[str] = None,
    prompt: str = AGENT_PROMPT_V2_SHORT,
    retriever_url: Optional[str] = None
) -> list[dict]:
    """
    Performs inference using Hugging Face Transformers with an explicit
    decision module:
    - retrieval path
    - direct path
    """
    if isinstance(question, str):
        question = [question]

    # Initialize tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_id if tokenizer_id else model_id
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    output = []
    for q in tqdm(question):
        decision = decide_retrieval(q)

        if decision:
            result = inference_hf_single(
                q,
                model,
                tokenizer,
                prompt,
                retriever_url=retriever_url
            )
        else:
            result = inference_hf_single_direct(
                q,
                model,
                tokenizer,
                DIRECT_PROMPT
            )

        search_queries = extract_search_queries(result)

        has_search = len(search_queries) > 0
        final_answer = extract_final_answer(result)
        row = {
            "question": q,
            "decision": decision,
            "decision_type": "retrieve" if decision else "direct",
            "result": result,
            "has_search": has_search,
            "search_queries": search_queries,
            "search_count": len(search_queries),
            "final_answer": final_answer,
        }
        row.update(build_diagnostic_fields(decision, has_search, final_answer))
        output.append(row)

    return output



def build_diagnostic_fields(decision: bool, has_search: bool, final_answer: Optional[str]) -> dict:
    final_answer_norm = final_answer.strip() if isinstance(final_answer, str) else ""
    return {
        "expected_search": decision,
        "actual_search": has_search,
        "search_mismatch": decision and not has_search,
        "search_overuse": (not decision) and has_search,
        "decision_search_consistent": decision == has_search,
        "answer_missing": not final_answer_norm,
    }

def inference_vllm_single(
    question: str,
    vllm_client: OpenAI,
    model_id: str,
    prompt=AGENT_PROMPT_V2_SHORT,
    tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
    retriever_url: Optional[str] = None,
):
    """
    Performs inference using an agentic RAG LLM with the specified XML format using vllm server with OpenAI API client.
    
    Args:
        question: The user's question to answer
        client: The OpenAI API client
        prompt: The prompt to use for the inference
        
    Returns:
        The final answer extracted from the model's response
    """
    # Format question
    question = question.strip()
    if question[-1] != '?':
        question += '?'

    # Stop string lists for different tags (match HF version)
    stop_on_search_list = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stop_on_answer_list = ["</answer>", " </answer>", "</answer>\n", " </answer>\n"]
    all_stop_strings = stop_on_search_list + stop_on_answer_list

    # Prepare the initial input (prompt and question only) and start think/step/reasoning
    input_text = f"{prompt}\nUser Question: {question}"
    think_step_reasoning = "\n<think>\n<step>\n    <reasoning>"
    # Apply chat template if a tokenizer is provided (to mirror HF tokenization formatting)
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": input_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        full_response = prompt + think_step_reasoning
    else:
        full_response = input_text + think_step_reasoning

    # Main inference loop
    step_count = 0
    max_steps = 10

    while step_count < max_steps:
        # Generate continuation from the current prefix using Completions API
        completion = vllm_client.completions.create(
            model=model_id,
            prompt=full_response,
            max_tokens=1024,
            stop=all_stop_strings,
            extra_body={"include_stop_str_in_output": True},
        )

        output_text = completion.choices[0].text

        # Append generated text to the running response
        full_response += output_text

        # If the model closed </answer> in this chunk, stop
        if "</answer>" in output_text:
            break

        # Check if we need to perform a search (based on just-generated text)
        search_query = extract_search_query(output_text)

        if search_query:  # Continue the generation with search results in context
            search_results = search(search_query, retriever_url=retriever_url)
            full_response += f"\n    <context>{search_results}</context>\n    <conclusion>"
        else:
            full_response += "\n    <conclusion>"

        step_count += 1

    print(f"Full response: {full_response}")
    return full_response


def inference_vllm(question: list[str] | str, api_key: str, base_url: str, model_id: str, tokenizer_id: Optional[str] = None, prompt=AGENT_PROMPT_V2_SHORT, retriever_url: Optional[str] = None):
    """
    Performs inference using an agentic RAG LLM with the specified XML format using vllm server with OpenAI API client.
    
    Args:
        question: The user's question to answer
        api_key: The API key for OpenAI
        base_url: The base URL for OpenAI
        prompt: The prompt to use for the inference
        
    Returns:
        The final answer extracted from the model's response
    """
    vllm_client = OpenAI(api_key=api_key, base_url=base_url)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id if tokenizer_id else model_id)
    if isinstance(question, str):
        question = [question]
    output = []
    for q in tqdm(question):
        output.append(inference_vllm_single(q, vllm_client, model_id, prompt, tokenizer, retriever_url=retriever_url))
    return output
    

# Async variants using AsyncOpenAI and search_async
async def inference_vllm_single_async(question: str, vllm_client: AsyncOpenAI, model_id: str, prompt=AGENT_PROMPT_V2_SHORT, tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None, retriever_url: Optional[str] = None) -> str:
    """
    Async version of `inference_vllm_single` using AsyncOpenAI and `search_async`.
    Mirrors the exact logic of the sync version.
    """
    # Format question
    question = question.strip()
    if question[-1] != '?':
        question += '?'

    # Stop string lists for different tags (match HF version)
    stop_on_search_list = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stop_on_answer_list = ["</answer>", " </answer>", "</answer>\n", " </answer>\n"]
    all_stop_strings = stop_on_search_list + stop_on_answer_list

    # Prepare the initial input (prompt and question only) and start think/step/reasoning
    input_text = f"{prompt}\nUser Question: {question}"
    think_step_reasoning = "\n<think>\n<step>\n    <reasoning>"
    # Apply chat template if a tokenizer is provided (to mirror HF tokenization formatting)
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": input_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        full_response = prompt + think_step_reasoning
    else:
        full_response = input_text + think_step_reasoning

    # Main inference loop
    step_count = 0
    max_steps = 10

    while step_count < max_steps:
        # Generate continuation from the current prefix using Completions API
        completion = await vllm_client.completions.create(
            model=model_id,
            prompt=full_response,
            max_tokens=1024,
            stop=all_stop_strings,
            extra_body={"include_stop_str_in_output": True},
        )

        output_text = completion.choices[0].text

        # Append generated text to the running response
        full_response += output_text

        # If the model closed </answer> in this chunk, stop
        if "</answer>" in output_text:
            break

        # Check if we need to perform a search (based on just-generated text)
        search_query = extract_search_query(output_text)

        if search_query:  # Continue the generation with search results in context
            search_results = await search_async(search_query, retriever_url=retriever_url)
            full_response += f"\n    <context>{search_results}</context>\n    <conclusion>"
        else:
            full_response += "\n    <conclusion>"

        step_count += 1

    # print(f"Full response: {full_response}")
    return full_response


async def inference_vllm_async(question: list[str] | str, api_key: str, base_url: str, model_id: str, tokenizer_id: Optional[str] = None, prompt=AGENT_PROMPT_V2_SHORT, max_concurrency: int = 64, retriever_url: Optional[str] = None) -> list[str]:
    """
    Async version of `inference_vllm` using AsyncOpenAI and `inference_vllm_single_async`.
    Mirrors the exact logic of the sync version.
    Note: if the output is weird, try to disable cascade attention when launching the vllm server.
    """
    vllm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id if tokenizer_id else model_id)
    if isinstance(question, str):
        question = [question]
    # Bounded concurrency using a semaphore to avoid overloading backend services
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(q: str) -> str:
        async with semaphore:
            return await inference_vllm_single_async(q, vllm_client, model_id, prompt, tokenizer, retriever_url=retriever_url)

    tasks = [run_one(q) for q in question]
    results = await tqdm_async.gather(*tasks)
    # Close the async search client
    await close_httpx_client_async()
    return list(results)


def inference():
    parser = argparse.ArgumentParser(description="Run decision-aware HF inference for Agentic RAG.")
    parser.add_argument("--input_jsonl", default="results/test_template.jsonl")
    parser.add_argument("--output_jsonl", default="results/hf_test_output.jsonl")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--tokenizer_id", default=None, help="Defaults to --model_id when omitted.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    args = parser.parse_args()

    global RETRIEVER_URL
    RETRIEVER_URL = args.retriever_url

    data_list = load_jsonl(args.input_jsonl)
    if args.max_samples is not None:
        data_list = data_list[:args.max_samples]
    questions = [data["question"] for data in data_list]

    tokenizer_id = args.tokenizer_id if args.tokenizer_id else args.model_id
    results = inference_hf(
        question=questions,
        model_id=args.model_id,
        tokenizer_id=tokenizer_id,
        prompt=AGENT_PROMPT_V2_SHORT,
        retriever_url=args.retriever_url,
    )

    for data, result in zip(data_list, results):
        data.update(result)

    write_jsonl(data_list, args.output_jsonl)
    print(f"Wrote {len(data_list)} rows with result to: {args.output_jsonl}")

# Example usage
if __name__ == "__main__":
    inference()    
