# LoRA Fine-tuning on SmolLM: Parameter-Efficient Adaptation

A complete implementation of Low-Rank Adaptation (LoRA) for efficient fine-tuning of SmolLM, achieving comparable performance to full fine-tuning with only **0.24% of parameters being trainable**.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-orange.svg)
![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-Latest-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project demonstrates **Parameter-Efficient Fine-Tuning (PEFT)** using LoRA (Low-Rank Adaptation), one of the most important techniques in modern large language model adaptation. Instead of fine-tuning all parameters, LoRA introduces small trainable rank decomposition matrices to selected layers.

### Key Achievements
- **99.76% Parameter Reduction**: Only 322K out of 134M parameters trainable
- **Comparable Performance**: Achieves similar or better perplexity than full fine-tuning
- **Memory Efficient**: Dramatically reduced memory requirements during training
- **Fast Training**: Significantly faster training compared to full fine-tuning
- **Practical Implementation**: Real-world applicable LoRA system

### LoRA Mathematical Foundation

For each targeted weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA introduces:
- Two low-rank matrices: $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$
- Where $r \ll \min(d, k)$ (rank is much smaller than original dimensions)

The adapted weight becomes:
$$W' = W + \frac{\alpha}{r}BA$$

Where:
- $W$ is the original frozen weight matrix
- $\alpha$ is a scaling hyperparameter  
- $r$ is the rank (bottleneck dimension)
- Only $A$ and $B$ are trained, dramatically reducing parameters

## Results Highlights

| Model Type | Trainable Parameters | Training Time | Perplexity (avg) |
|------------|---------------------|---------------|------------------|
| Full Fine-tuning | 134.5M (100%) | ~Hours | 27.35 |
| **LoRA Fine-tuning** | **322K (0.24%)** | **~Minutes** | **15.44** |

**Memory Savings**: ~99.76% reduction in trainable parameters
**Performance**: Better average perplexity than full fine-tuning

## Architecture

### SmolLM Base Model
- **Parameters**: 135M total parameters
- **Architecture**: Decoder-only Transformer with Grouped Query Attention
- **Components**: 30 layers, 9 attention heads, 3 KV heads
- **Techniques**: RMSNorm, SiLU activation, Rotary Position Embeddings

### LoRA Configuration
- **Target Modules**: Query, Key, Value projections in attention layers
- **Rank**: 4 (low-rank bottleneck dimension)
- **Scaling**: α=8 for optimal adaptation strength
- **Dropout**: 0.3 for regularization

## Usage

### Installation
```bash
cd lora-smollm-finetuning
pip install -r requirements.txt
```

### Running the Notebook
```bash
jupyter notebook lora_smollm_finetuning.ipynb
```

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [SmolLM: Small but Mighty Language Models](https://huggingface.co/HuggingFaceTB/SmolLM-135M)