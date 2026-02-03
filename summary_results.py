#!/usr/bin/env python3
"""
결과 요약 스크립트
"""

import json

def print_summary():
    with open('results/results_20260130_052621.json', 'r') as f:
        data = json.load(f)

    print("=" * 80)
    print("멀티모달 LLM 텍스트 우위성 분석 결과 요약")
    print("=" * 80)

    # 1. Image & Video 모달리티
    print("\n[1] Image & Video 모달리티")
    print("-" * 80)
    for modality in ['image', 'video']:
        if modality in data:
            result = data[modality]
            print(f"\n{modality.upper()} ({result['model']}):")
            print(f"  데이터셋: {result['dataset']}")
            for stage in ['early', 'middle', 'late']:
                metrics = result['metrics'][stage]
                print(f"  {stage:8}: MDI={metrics['mdi']:6.2f}, AEI={metrics['aei']:6.2f}")

    # 2. 토큰 스케일링 효과 (Audio, Time-Series, Graph)
    print("\n\n[2] 토큰 스케일링 효과")
    print("-" * 80)
    for modality in ['audio', 'timeseries', 'graph']:
        if modality in data:
            result = data[modality]
            print(f"\n{modality.upper()} ({result['model']}):")
            for factor in ['1', '5', '10']:
                if factor in result['scaling_results']:
                    metrics = result['scaling_results'][factor]
                    print(f"  x{factor:2}: MDI (early={metrics['mdi_early']:5.2f}, "
                          f"middle={metrics['mdi_middle']:6.2f}, "
                          f"late={metrics['mdi_late']:6.2f})")

    # 3. 토큰 압축 효과
    print("\n\n[3] 토큰 압축 효과 (FasterVLM)")
    print("-" * 80)
    if 'compression' in data:
        result = data['compression']
        print(f"\n모델: {result['model']}")
        print(f"방법: {result['method']}")
        for rate in ['0.0', '0.75', '0.9', '0.95']:
            if rate in result['compression_results']:
                metrics = result['compression_results'][rate]
                reduction = float(rate) * 100
                print(f"\n  {reduction:4.0f}% 압축:")
                print(f"    Early:  MDI={metrics['mdi_early']:5.2f}, AEI={metrics['aei_early']:5.2f}")
                print(f"    Middle: MDI={metrics['mdi_middle']:5.2f}, AEI={metrics['aei_middle']:5.2f}")
                print(f"    Late:   MDI={metrics['mdi_late']:5.2f}, AEI={metrics['aei_late']:5.2f}")

    # 4. 주요 발견 사항
    print("\n\n[4] 주요 발견 사항")
    print("-" * 80)
    print("\n✓ 텍스트 우위성은 보편적:")
    print("  - 모든 MLLM에서 MDI > 1 (텍스트 우위)")
    print("  - Image: MDI=17.37 (late layer)")
    print("  - Video: MDI=157.53 (late layer) - 매우 높은 텍스트 우위")

    print("\n✓ 레이어가 깊어질수록 우위성 증가:")
    print("  - Early → Middle → Late 레이어로 갈수록 MDI 증가")
    print("  - LLaVA: 1.58 → 10.23 → 17.37")

    print("\n✓ 토큰 복제가 우위성 증가:")
    print("  - Audio x1: MDI=1.16 → x10: MDI=8.70")
    print("  - Time-Series x1: MDI=3.52 → x10: MDI=16.25")

    print("\n✓ 토큰 압축으로 우위성 완화:")
    print("  - 90% 압축: MDI 17.37 → 1.84 (균형에 가까워짐)")
    print("  - FasterVLM 기법으로 텍스트 우위성 크게 감소")

    print("\n✓ Graph 모달리티는 초기에 비텍스트 우위:")
    print("  - x1: MDI=0.20 (< 1, 그래프 우위)")
    print("  - 하지만 토큰 복제 시 텍스트 우위로 전환")

    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    print_summary()
