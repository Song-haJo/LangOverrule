# Text Dominance Analysis in Multimodal Large Language Models

Re-implementation of **"When Language Overrules: Revealing Text Dominance in Multimodal Large Language Models"** (arXiv:2508.10552, August 2025)

## Overview

This repository provides tools to analyze and mitigate **text dominance** in Multimodal Large Language Models (MLLMs). Text dominance is the phenomenon where MLLMs over-rely on textual inputs while underutilizing information from other modalities (images, video, audio, time-series, graphs).

### Key Features

- **MDI (Modality Dominance Index)**: Quantifies relative attention per token between text and non-text modalities
- **AEI (Attention Efficiency Index)**: Measures attention efficiency relative to token proportion
- **Multi-modality support**: Image, Video, Audio, Time-Series, Graph
- **Token Compression**: [CLS]-based visual token pruning (FasterVLM) to mitigate text dominance

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/text-dominance-analysis.git
cd text-dominance-analysis

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- transformers 4.40+
- CUDA (recommended for GPU acceleration)

## 🚀 Quick Start with run.sh

### One-Command Execution

```bash
# Run tests and paper comparison
./run.sh test

# Run LLaVA with 10 dummy samples
./run.sh llava 10

# Run Qwen2.5-VL with 100 real MMMU Pro samples
./run.sh qwen 100 real

# Run both models
./run.sh both 50 real
```

## Quick Start (Python API)

### 1. Compute MDI and AEI Metrics

```python
from src.metrics import compute_modality_metrics, MDI, AEI
import torch

# Example: Compute metrics from attention weights
attention_weights = torch.randn(1, 12, 512, 512)  # (batch, heads, seq, seq)
text_mask = torch.zeros(512, dtype=torch.bool)
text_mask[100:200] = True  # Text tokens at positions 100-200
nontext_mask = torch.zeros(512, dtype=torch.bool)
nontext_mask[:100] = True  # Image tokens at positions 0-100

# Compute metrics
metrics = compute_modality_metrics(
    attention_weights,
    text_mask,
    nontext_mask
)
print(f"MDI: {metrics.mdi:.2f}")
print(f"AEI (text): {metrics.aei_text:.2f}")
```

### 2. Analyze LLaVA Model

```python
from src.models import LLaVAWrapper, ModelConfig
from PIL import Image

# Initialize model
config = ModelConfig(
    model_name="llava",
    model_path="llava-hf/llava-1.5-7b-hf",
    device="cuda"
)
wrapper = LLaVAWrapper(config)
wrapper.load_model()

# Analyze image-text sample
result = wrapper.forward_with_attention(
    text="Describe this image in detail.",
    image=Image.open("example.jpg")
)

print(f"MDI (late layers): {result.metrics['late'].mdi:.2f}")
```

### 3. Apply Token Compression

```python
from src.compression import CLSTokenPruner, FasterVLM

# Initialize pruner with 90% reduction
pruner = CLSTokenPruner(reduction_rate=0.90)

# Prune visual tokens
result = pruner.prune(visual_tokens, attention_weights)
print(f"Retained {result.retained_count}/{result.original_count} tokens")
```

### 4. Run Full Analysis

```bash
# Analyze all modalities
python experiments/run_analysis.py --modality all --output-dir results

# Analyze specific modality
python experiments/run_analysis.py --modality image

# Include token compression experiments
python experiments/run_analysis.py --modality image --run-compression
```

## Metrics

### Modality Dominance Index (MDI)

MDI measures relative attention per token between modalities:

```
MDI = (A_T / |T|) / (A_O / |O|)
```

Where:
- `A_T`: Total attention to text tokens
- `A_O`: Total attention to non-text tokens
- `|T|`, `|O|`: Token counts

**Interpretation:**
- MDI > 1: Text dominance
- MDI < 1: Non-text dominance
- MDI ≈ 1: Balanced

### Attention Efficiency Index (AEI)

AEI measures attention efficiency relative to token proportion:

```
AEI_T = P_T / Q_T
```

Where:
- `P_T = A_T / (A_T + A_O)`: Attention proportion
- `Q_T = |T| / (|T| + |O|)`: Token proportion

## Supported Models

| Modality | Model | Status |
|----------|-------|--------|
| Image | LLaVA-1.5-7B/13B | ✅ |
| Image | Qwen2.5-VL | ✅ |
| Image | Kimi-VL | ✅ |
| Video | VideoLLaMA2/3 | ✅ |
| Audio | Qwen2-Audio | ✅ |
| Time-Series | ChatTS | ✅ |
| Graph | GraphGPT | ✅ |

## Key Findings (from Paper)

1. **Text dominance is pervasive**: All tested MLLMs show significant text dominance (MDI >> 1)

2. **Deeper layers amplify dominance**: MDI increases in later transformer layers
   - LLaVA-1.5-7B: Early=1.58, Late=17.37

3. **Token redundancy drives attention dilution**: Non-text modalities have many redundant tokens

4. **Token compression mitigates dominance**: 90% reduction brings MDI from 17.37 to 1.84

## Citation

```bibtex
@article{wu2025when,
  title={When Language Overrules: Revealing Text Dominance in Multimodal Large Language Models},
  author={Wu, Huyu and Tang, Meng and Zheng, Xinhan and Jiang, Haiyun},
  journal={arXiv preprint arXiv:2508.10552},
  year={2025}
}
```

## License

MIT License


## Project 구조

```
c:\overrule\
├── src/
│   ├── metrics/           # MDI/AEI 메트릭 구현
│   │   ├── mdi.py         # Modality Dominance Index
│   │   ├── aei.py         # Attention Efficiency Index
│   │   └── combined.py    # 통합 메트릭 계산
│   ├── attention/         # 어텐션 추출/분석
│   │   ├── extractor.py   # 어텐션 가중치 추출
│   │   └── analyzer.py    # 크로스모달 어텐션 분석
│   ├── models/            # 모델 래퍼
│   │   ├── llava.py       # 이미지 (LLaVA)
│   │   ├── video_llama.py # 비디오 (VideoLLaMA)
│   │   ├── qwen_audio.py  # 오디오 (Qwen-Audio)
│   │   ├── timeseries.py  # 시계열 (ChatTS)
│   │   └── graph.py       # 그래프 (GraphGPT)
│   ├── compression/       # 토큰 압축
│   │   └── token_pruning.py  # [CLS] 기반 FasterVLM
│   └── utils/
│       └── visualization.py  # 시각화 (논문 Figure 재현)
├── experiments/
│   └── run_analysis.py    # 실험 실행 스크립트
├── configs/
│   └── default.yaml       # 설정 파일
├── requirements.txt
└── README.md
```

# 핵심 구현 내용
| 구성요소	| 파일 	| 설명 |
|-|-|-|
|MDI 메트릭	| mdi.py	| 수식 (1) 구현 |
| AEI 메트릭	| aei.py	| 수식 (2)-(4) 구현 |
| 토큰 압축	| token_pruning.py	| 수식 (5)-(8) 구현 |
|5개 모달리티	|models/	|이미지, 비디오, 오디오, 시계열, 그래프|

실제 모델로 실험하려면 HuggingFace에서 모델을 다운로드해야 합니다 (예: llava-hf/llava-1.5-7b-hf).