#!/usr/bin/env python3
"""
간단한 LLaVA 모델 로드 테스트
"""

import os
import sys
import torch

print("="*80)
print("Step 1: 환경 확인")
print("="*80)

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print(f"\nHF_HOME: {os.getenv('HF_HOME', 'Not set')}")
print(f"HF_TOKEN: {'Set' if os.getenv('HF_TOKEN') else 'Not set'}")

print("\n" + "="*80)
print("Step 2: Transformers 확인")
print("="*80)

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"Transformers import error: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("Step 3: LLaVA 모델 다운로드 (이 단계가 오래 걸릴 수 있습니다)")
print("="*80)

model_path = "llava-hf/llava-1.5-7b-hf"
print(f"Model: {model_path}")
print("다운로드 시작...")

try:
    from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    # 4-bit 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("모델 로드 중 (4-bit 양자화)...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )

    print("✓ 모델 로드 완료!")
    print(f"모델 타입: {type(model)}")
    print(f"모델 device: {next(model.parameters()).device}")

    print("\nProcessor 로드 중...")
    processor = AutoProcessor.from_pretrained(model_path)
    print("✓ Processor 로드 완료!")

    print("\n" + "="*80)
    print("Step 4: 간단한 Forward Pass 테스트")
    print("="*80)

    # 더미 입력 생성
    print("더미 입력 생성...")
    from PIL import Image
    import numpy as np

    # 더미 이미지 (100x100)
    dummy_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    dummy_text = "<image>\nWhat is in this image?"

    print("입력 처리...")
    inputs = processor(text=dummy_text, images=dummy_image, return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print("Forward pass...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    print(f"✓ Forward pass 완료!")
    print(f"  출력 타입: {type(outputs)}")
    print(f"  Logits shape: {outputs.logits.shape}")

    if hasattr(outputs, 'attentions') and outputs.attentions:
        print(f"  Attention 레이어 수: {len(outputs.attentions)}")
        print(f"  첫 번째 attention shape: {outputs.attentions[0].shape}")

    print("\n" + "="*80)
    print("모든 테스트 성공!")
    print("="*80)
    print("\n이제 test_real_model.py를 실행할 수 있습니다.")

except Exception as e:
    print(f"\n오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
