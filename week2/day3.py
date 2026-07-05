"""
Week 2 - Day 3 : 엣지 검출 기법 비교 + 전처리 파이프라인 통합
AI허브 주행 프레임을 대상으로 여러 엣지 검출 기법을 비교하고,
Canny 기반 전처리 파이프라인을 하나의 함수로 통합한다.
사다리꼴 ROI까지 적용해 3주차 Hough Transform 입력을 준비한다.

실행 방법:
    python week2_day3_edge_pipeline.py

각 단계는 함수로 분리되어 있고, main에서 순서대로 호출한다.
"""

import cv2
import glob
import os
import numpy as np

# ─────────────────────────────────────────────
# 설정 (여기만 네 환경에 맞게 수정)
# ─────────────────────────────────────────────
FRAME_DIR = "도로주행영상/Validation/원천데이터/bb/자동차전용도로/주간일출/맑음/30_전방/20200910_104924/1/"       # 프레임 이미지 폴더
FRAME_EXT = "*.jpg"              # 확장자 (.png 이면 "*.png")
BLUR_KSIZE = 9                   # Gaussian Blur 커널 크기 (홀수)

# 2주차 Day2 트랙바에서 찾은 최적 Canny 값
BEST_LOW = 50
BEST_HIGH = 100

OUTPUT_DIR = "results_week2"     # 결과 저장 폴더


def load_frame_paths():
    """폴더에서 프레임 이미지 경로를 정렬해 가져온다."""
    paths = sorted(glob.glob(os.path.join(FRAME_DIR, FRAME_EXT)))
    print(f"[load] 찾은 이미지 수: {len(paths)}")
    if len(paths) == 0:
        print("[load] 경고: 이미지를 못 찾음. FRAME_DIR / FRAME_EXT 를 확인하세요.")
    return paths


# ─────────────────────────────────────────────
# 단계 1 : 엣지 검출 기법 비교
#          Sobel / Sobel Magnitude / Laplacian / Canny 를 나란히 본다.
# ─────────────────────────────────────────────
def step1_compare_methods(paths):
    print("\n[단계1] 엣지 검출 기법 비교 (Sobel / Magnitude / Laplacian / Canny)")
    img = cv2.imread(paths[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

    # Sobel X, Y (방향별 1차 미분)
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

    # Sobel Magnitude (X,Y 합침 → 방향 무관 전체 엣지 강도)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)

    # Laplacian (2차 미분, 노이즈에 매우 민감)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)

    # Canny (2주차 최적값)
    canny = cv2.Canny(blur, BEST_LOW, BEST_HIGH)

    # float(CV_64F) 결과는 convertScaleAbs 로 절댓값+uint8 변환해야 제대로 보임
    cv2.imshow("1. sobel_x", cv2.convertScaleAbs(sobel_x))
    cv2.imshow("2. sobel_y", cv2.convertScaleAbs(sobel_y))
    cv2.imshow("3. sobel_magnitude", cv2.convertScaleAbs(sobel_mag))
    cv2.imshow("4. laplacian", cv2.convertScaleAbs(laplacian))
    cv2.imshow("5. canny", canny)

    print("        아무 키나 누르면 다음 단계로.")
    print("        관찰 포인트: Canny가 가장 얇고 깔끔한 이진 엣지를 낸다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# 단계 3 : 전처리 파이프라인 (통합 함수)
#          gray → blur → Canny → 직사각형 ROI
#          파라미터를 인자로 받아 데이터에 맞게 조절 가능.
# ─────────────────────────────────────────────
def preprocess(frame, low=BEST_LOW, high=BEST_HIGH, blur_k=BLUR_KSIZE):
    """주행 프레임 → 차선 인식용 엣지 이미지 (전체)"""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    edges = cv2.Canny(blur, low, high)
    return edges


# ─────────────────────────────────────────────
# 단계 4 : 사다리꼴 ROI
#          원근을 고려해 도로 영역(사다리꼴)만 남긴다.
#          직사각형 ROI보다 잡선 제거에 효과적.
# ─────────────────────────────────────────────
def region_of_interest(edges):
    h, w = edges.shape[:2]
    mask = np.zeros_like(edges)              # 검정 바탕 마스크

    # 도로 모양 사다리꼴 (좌표는 영상에 맞게 조정)
    polygon = np.array([[
        (int(0), h),                  # 좌하단
        (int(w * 0.35), int(h * 0.60)),      # 좌상단 (소실점 근처)
        (int(w * 0.65), int(h * 0.60)),      # 우상단
        (int(w), h),                  # 우하단
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)         # 사다리꼴 내부만 흰색
    masked = cv2.bitwise_and(edges, mask)    # 엣지 ∩ 마스크 = 도로 영역 엣지만
    return masked


# ─────────────────────────────────────────────
# 단계 2 : 기법 비교 결과 분석 (텍스트 출력)
#          단계1 관찰을 표로 정리해 콘솔에 남긴다.
# ─────────────────────────────────────────────
def step2_print_analysis():
    print("\n[단계2] 기법별 성능 비교 분석")
    print("  ┌────────────────┬──────────────────────────┬────────┬───────────┐")
    print("  │ 기법           │ 특징                     │ 노이즈 │ 차선 적합  │")
    print("  ├────────────────┼──────────────────────────┼────────┼───────────┤")
    print("  │ Sobel X        │ 세로선 위주 (차선 유리)  │ 많음   │ 중간      │")
    print("  │ Sobel Y        │ 가로선 위주              │ 많음   │ 낮음      │")
    print("  │ Sobel Magnitude│ 모든 방향, 두꺼운 엣지   │ 많음   │ 중간      │")
    print("  │ Laplacian      │ 2차 미분, 매우 예민      │ 아주많음│ 낮음     │")
    print("  │ Canny          │ 얇고 깔끔, 이진, 연결성  │ 적음   │ 높음★    │")
    print("  └────────────────┴──────────────────────────┴────────┴───────────┘")
    print("  결론: Canny는 히스테리시스(low/high 연결성)로 깔끔한 이진 선을")
    print("        내고 노이즈를 걸러 차선 인식의 표준으로 채택.")


# ─────────────────────────────────────────────
# 단계 5 : 여러 프레임에 최종 파이프라인 적용 + 저장
# ─────────────────────────────────────────────
def step5_apply_and_save(paths, n=10, save_n=3, delay_ms=500):
    print(f"\n[단계5] 앞 {n}개 프레임에 최종 파이프라인 적용, {save_n}장 저장")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, path in enumerate(paths[:n]):
        frame = cv2.imread(path)
        edges = preprocess(frame)            # 단계3
        roi = region_of_interest(edges)      # 단계4 (사다리꼴)

        cv2.imshow("final pipeline", roi)

        if i < save_n:
            # 원본 + 최종 결과를 나란히 저장 (비교용)
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            combined = cv2.hconcat([frame, roi_bgr])
            out_path = os.path.join(OUTPUT_DIR, f"final_{i}.jpg")
            cv2.imwrite(out_path, combined)
            print(f"        저장: {out_path}")

        if cv2.waitKey(delay_ms) == ord('q'):
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# main : 단계별 순차 실행
#        필요 없는 단계는 주석 처리하면 됨
# ─────────────────────────────────────────────
if __name__ == "__main__":
    paths = load_frame_paths()

    if len(paths) > 0:
        step1_compare_methods(paths)   # 기법 비교 (시각)
        step2_print_analysis()         # 비교 분석 (표)
        step5_apply_and_save(paths)    # 통합 파이프라인 적용 + 저장
        # preprocess(), region_of_interest() 는 step5 내부에서 사용됨

    print("\n[done] 종료")