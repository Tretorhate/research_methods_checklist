# AI Ethics Checklist Validation Study
This repository contains code for evaluating the effectiveness of AI ethics checklists in reducing bias in language model responses.

## Overview

The study compares baseline model responses against responses generated with ethical prompting based on a reference-grounded checklist. It measures bias reduction across multiple scenarios and models using statistical analysis.

## Requirements

### Python Dependencies

```bash
pip install ollama numpy scipy
```

### Ollama Installation

Install Ollama service:

```bash
winget install Ollama.Ollama
```

### Model Setup

Pull the required models:

```bash
ollama pull gemma3:1b
ollama pull qwen3:0.6b
ollama pull deepseek-r1:1.5b
```

## Usage

### Run the main study

```bash
python checklist_v2.py
```

### Run the basic version

```bash
python checklist.py
```

## Output

The script generates:

- Bias scores for baseline and checklist-guided responses
- Statistical analysis with paired t-test
- Per-model performance breakdown
- Hypothesis confirmation based on 20% bias reduction threshold

## Methodology

The study uses:

- 7 ethical criteria mapped to academic sources
- 5 deployment-stage scenarios covering healthcare, hiring, climate policy, elderly care, and education
- Multi-dimensional bias detection with keyword analysis
- Paired statistical testing for significance

## Limitations

- Simplified bias metrics based on keyword detection
- Small sample size (3 models x 5 scenarios)
- Requires manual expert review for comprehensive validation

## Next Steps

For production use, consider:

- Integrating fairlearn for standard fairness metrics
- Expanding to more models and scenarios
- Adding human evaluation surveys
- Implementing continuous monitoring
