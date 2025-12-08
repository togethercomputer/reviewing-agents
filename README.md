<h1 align="center">Reviewing Agents</h1>

<p align="center">
  <img src="assets/logo.png" alt="Reviewing Agents" width="400">
</p>

<p align="center">
  <i>LLM-powered scientific paper review and error detection</i>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2511.15534-b31b1b.svg)](https://arxiv.org/abs/2511.15534)
[![arXiv](https://img.shields.io/badge/arXiv-2512.05925-b31b1b.svg)](https://arxiv.org/abs/2512.05925)
[![Conference](https://img.shields.io/badge/Agents4Science-2025-blue.svg)](https://agents4science.stanford.edu/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

This repository supports:

- [**To Err Is Human**](https://arxiv.org/abs/2512.05925): Systematic Quantification of Errors in Published AI Papers via LLM Analysis
  
  We developed an LLM-based Paper Correctness Checker to identify objective mistakes (formulas, derivations, figures, tables) in papers published at top AI venues. Our analysis reveals that mistakes per paper have increased over time—from 3.8 in NeurIPS 2021 to 5.9 in NeurIPS 2025 (+55%). Human experts confirmed 83.2% precision on 316 reviewed mistakes. The checker can also propose correct fixes for 75.8% of identified issues.

- [**Agents4Science**](https://arxiv.org/abs/2511.15534): LLM reviewers for the [Agents4Science Conference](https://agents4science.stanford.edu/), the first conference where AI agents served as both primary authors and reviewers
  
  Agents4Science 2025 was the inaugural conference where AI systems served as both authors and reviewers of research papers. Organized by TogetherAI and Stanford University, it received 315 submissions with 48 papers accepted after AI + human peer review. 

---

## Installation

```bash
uv sync
```

---

## Configuration

Set up API keys in `.env` (copy from `env.dev`):

```bash
cp env.dev .env
```

Configure modules in `config.yaml`.

---

## Modules

| Module | Description |
|--------|-------------|
| `SimpleReviewer` | General paper reviewing |
| `LLMCorrectnessDetector` | Methodological correctness evaluation |
| `LLMCriticalityVerifier` | Verifies criticality of correctness findings |
| `LLMFormatDetector` | Format compliance checking |
| `JailbreakingChecker` | Detects adversarial instructions in papers |
| `ReferenceCheckLight` | Reference hallucination detection |
| `ReferenceCheckHeavy` | Full reference + author verification |
| `ArxivTaxonomyClassifier` | arXiv category classification |

---

## Quick Start

[Agents4Science](https://arxiv.org/abs/2511.15534) reviewers can be used to review papers:

```python
from reviewing_agents.modules import SimpleReviewer

pdf_bytes = open("paper.pdf", "rb").read()

reviewer = SimpleReviewer()
result = reviewer.review_paper(pdf_bytes)
```

The `LLMCorrectnessDetector` and `LLMCriticalityVerifier` can be used to evaluate the correctness of a paper, used in our [To Err Is Human](https://arxiv.org/abs/2512.05925) paper.

```python
from reviewing_agents.modules import LLMCorrectnessDetector, LLMCriticalityVerifier

pdf_bytes = open("paper.pdf", "rb").read()

detector = LLMCorrectnessDetector()
correctness = detector.check_correctness(pdf_bytes)

verifier = LLMCriticalityVerifier()
findings = {"score": correctness.score, "reasoning": correctness.reasoning, "key_issues": correctness.key_issues}
verified = verifier.verify_criticality(pdf_bytes, findings)
```

---

## Citation

```bibtex
@article{bianchi2025toerr,
  title={To Err Is Human: Systematic Quantification of Errors in Published AI Papers via LLM Analysis},
  author={Bianchi, Federico and Kwon, Yongchan and Izzo, Zachary and Zhang, Linjun and Zou, James},
  journal={arXiv preprint arXiv:2512.05925},
  year={2025}
}

@article{bianchi2025agents4science,
  title={Exploring the use of AI authors and reviewers at Agents4Science},
  author={Bianchi, Federico and Queen, Owen and Thakkar, Nitya and Sun, Eric and Zou, James},
  journal={arXiv preprint arXiv:2511.15534},
  year={2025}
}
```
