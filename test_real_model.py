#!/usr/bin/env python3
"""
실제 LLaVA 모델을 로드하여 텍스트 우위성 분석 테스트

이 스크립트는 HuggingFace에서 LLaVA-1.5-7B 모델을 다운로드하고,
실제 이미지를 사용하여 MDI/AEI를 계산합니다.
"""

import os
import sys
import torch
from PIL import Image
import requests
from io import BytesIO

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.models.llava import LLaVAAnalyzer
from src.metrics import compute_modality_metrics


def download_sample_image(url: str = None) -> Image.Image:
    """
    샘플 이미지 다운로드

    Args:
        url: 이미지 URL (기본값: COCO 샘플 이미지)

    Returns:
        PIL Image
    """
    if url is None:
        # COCO 데이터셋의 샘플 이미지
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    print(f"다운로드 중: {url}")
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    print(f"이미지 크기: {image.size}")
    return image


def test_llava_real_model():
    """LLaVA 실제 모델 테스트"""

    print("="*80)
    print("실제 LLaVA 모델로 텍스트 우위성 분석 테스트")
    print("="*80)

    # GPU 확인
    print(f"\nGPU 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 장치: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 모델 경로 설정
    model_path = "llava-hf/llava-1.5-7b-hf"

    print(f"\n{'='*80}")
    print(f"모델 로드 중: {model_path}")
    print(f"{'='*80}")
    print("주의: 처음 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다 (~14GB)")

    # 환경 변수에서 HF_TOKEN 확인
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print(f"HuggingFace 토큰 발견 (길이: {len(hf_token)})")

    # Analyzer 초기화 (4-bit 양자화 사용하여 메모리 절약)
    print("\nAnalyzer 초기화 중 (4-bit 양자화)...")
    analyzer = LLaVAAnalyzer(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit=True,  # 메모리 절약
    )

    # 모델 로드
    print("모델 로드 중...")
    analyzer.wrapper.load_model()
    print("✓ 모델 로드 완료!")

    # 샘플 이미지 다운로드
    print(f"\n{'='*80}")
    print("샘플 이미지 준비")
    print(f"{'='*80}")
    image = download_sample_image()

    # 테스트 프롬프트
    text_prompt = "Describe this image in detail."
    print(f"\n프롬프트: '{text_prompt}'")

    # 분석 실행
    print(f"\n{'='*80}")
    print("텍스트 우위성 분석 실행")
    print(f"{'='*80}")

    result = analyzer.analyze_sample(text_prompt, image)

    # 결과 출력
    print(f"\n{'='*80}")
    print("분석 결과")
    print(f"{'='*80}")

    print(f"\n레이어 수: {result['num_layers']}")
    print(f"텍스트 토큰 수: {result['token_masks'].num_text_tokens}")
    print(f"이미지 토큰 수: {result['token_masks'].num_nontext_tokens}")

    print(f"\n{'='*80}")
    print("MDI & AEI 메트릭")
    print(f"{'='*80}")

    if result['metrics']:
        for stage in ['early', 'middle', 'late']:
            if stage in result['metrics']:
                m = result['metrics'][stage]
                print(f"\n{stage.upper()} layers:")
                print(f"  MDI (Modality Dominance Index): {m.mdi:.4f}")
                print(f"  AEI (Attention Efficiency Index): {m.aei_text:.4f}")
                print(f"  Text attention: {m.text_attention:.4f}")
                print(f"  Image attention: {m.nontext_attention:.4f}")

                if m.mdi > 1:
                    print(f"  → 텍스트 우위 (텍스트가 이미지보다 {m.mdi:.2f}배 더 주목)")
                elif m.mdi < 1:
                    print(f"  → 이미지 우위 (이미지가 텍스트보다 {1/m.mdi:.2f}배 더 주목)")
                else:
                    print(f"  → 균형 잡힌 주목")

    # 논문 결과와 비교
    print(f"\n{'='*80}")
    print("논문 결과와 비교 (LLaVA-1.5-7B, Table 1)")
    print(f"{'='*80}")

    paper_results = {
        'early': {'mdi': 1.58, 'aei': 1.04},
        'middle': {'mdi': 10.23, 'aei': 3.51},
        'late': {'mdi': 17.37, 'aei': 4.23}
    }

    print("\nStage     | Paper (MDI, AEI) | Actual (MDI, AEI) | Diff")
    print("-" * 70)

    for stage in ['early', 'middle', 'late']:
        if stage in result['metrics']:
            p = paper_results[stage]
            a = result['metrics'][stage]
            mdi_diff = abs(a.mdi - p['mdi'])
            aei_diff = abs(a.aei_text - p['aei'])
            print(f"{stage:8}  | ({p['mdi']:6.2f}, {p['aei']:5.2f})  | "
                  f"({a.mdi:6.2f}, {a.aei_text:5.2f})  | "
                  f"Δ MDI={mdi_diff:5.2f}, AEI={aei_diff:5.2f}")

    print(f"\n{'='*80}")
    print("테스트 완료!")
    print(f"{'='*80}")

    print("\n참고:")
    print("- 논문과 정확히 일치하지 않을 수 있습니다 (데이터셋 차이)")
    print("- 논문은 MMMU Pro 벤치마크 100개 샘플의 평균")
    print("- 이 테스트는 단일 이미지로 실행")
    print("- 경향성 (early < middle < late)은 유사해야 함")


def test_multiple_samples():
    """여러 샘플로 테스트 (논문과 유사하게)"""

    print(f"\n{'='*80}")
    print("여러 샘플로 테스트")
    print(f"{'='*80}")

    # 다양한 이미지 URL
    sample_images = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # 고양이
        "http://images.cocodataset.org/val2017/000000397133.jpg",  # 식사
        "http://images.cocodataset.org/val2017/000000252219.jpg",  # 거리
    ]

    prompts = [
        "What animals are in this image?",
        "Describe what you see in this image.",
        "What is happening in this scene?",
    ]

    # Analyzer 초기화
    analyzer = LLaVAAnalyzer(
        model_path="llava-hf/llava-1.5-7b-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit=True,
    )

    samples = []
    for img_url, prompt in zip(sample_images, prompts):
        try:
            image = download_sample_image(img_url)
            samples.append({'text': prompt, 'image': image})
        except Exception as e:
            print(f"이미지 로드 실패 ({img_url}): {e}")

    if samples:
        print(f"\n{len(samples)}개 샘플 분석 중...")
        results = analyzer.analyze_dataset(samples, progress=True)

        print(f"\n{'='*80}")
        print("평균 결과")
        print(f"{'='*80}")

        for stage, metrics in results.items():
            print(f"\n{stage.upper()} layers:")
            print(f"  MDI: {metrics['mdi_mean']:.2f} ± {metrics['mdi_std']:.2f}")
            print(f"  AEI: {metrics['aei_mean']:.2f} ± {metrics['aei_std']:.2f}")
            print(f"  샘플 수: {metrics['num_samples']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLaVA 실제 모델 테스트")
    parser.add_argument(
        "--multiple",
        action="store_true",
        help="여러 샘플로 테스트"
    )

    args = parser.parse_args()

    try:
        if args.multiple:
            test_multiple_samples()
        else:
            test_llava_real_model()
    except KeyboardInterrupt:
        print("\n\n중단됨")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
