# Agentic_RAG

This repository is based on [HiPRAG](https://github.com/qualidea1217/HiPRAG) and is being adapted for my undergraduate thesis project on Agentic RAG strategy optimization.

The current goal is to build a decision-aware Agentic RAG baseline, and then further explore token-level action decomposition for retrieval-related actions.

## Upstream Project

This project is derived from HiPRAG, which is licensed under Apache License 2.0.

Original paper:
- HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation

Original resources:
- Code: https://github.com/qualidea1217/HiPRAG
- Paper: https://arxiv.org/abs/2510.07794

## Current Project Status

Compared with the original HiPRAG codebase, the current repository has already introduced several modifications for decision-aware Agentic RAG experimentation:

### 1. Pre-decision module
A `decide_retrieval(question)` function is added before inference to decide whether a question should enter the retrieval path or the direct path.

### 2. Dual-path inference
The inference pipeline is split into:
- retrieval path
- direct path

The retrieval path preserves the original search-augmented generation behavior, while the direct path answers without triggering retrieval.

### 3. Direct prompt
A dedicated `DIRECT_PROMPT` is added for direct answering without `<search>`.

### 4. Structured output logging
The inference output is expanded from plain text to structured fields including:

- `decision`
- `decision_type`
- `result`
- `has_search`
- `search_queries`
- `search_count`
- `final_answer`

### 5. Cleaner response parsing
Additional helper functions are added to:
- strip assistant output from chat template prefixes
- extract search queries
- extract final answers from `<answer>...</answer>`

## Minimal Run

### 1. Start retrieval server

```bash
conda activate retriever
bash retrieval_launch.sh
```

### 2. Run inference

```bash
python inference.py
```

## Default Input / Output

Default input:
- `results/test_template.jsonl`

Default output:
- `results/hf_test_output.jsonl`

## Notes

This repository is an experimental development version for Agentic RAG research, rather than an exact mirror of the original HiPRAG repository.

The current codebase is primarily used for:
- running and understanding the baseline
- adding explicit retrieval decision logic
- preparing for later token-level action decomposition experiments

## Recommended Environment

Current commonly used environment:

```bash
conda activate retriever
```

## Acknowledgments

This project builds upon the excellent work of several open-source projects:

- HiPRAG
- Search-R1
- veRL
- DeepSeek-R1
- RAGEN

Sincere thanks to the original authors and contributors.

## Citation

If you use the original HiPRAG framework, please cite the original paper:

```bibtex
@misc{wu2025hipraghierarchicalprocessrewards,
      title={HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation}, 
      author={Peilin Wu and Mian Zhang and Kun Wan and Wentian Zhao and Kaiyu He and Xinya Du and Zhiyu Chen},
      year={2025},
      eprint={2510.07794},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.07794}
}
```

## License

This project follows the Apache License 2.0 inherited from the upstream HiPRAG project. See `LICENSE` for details.