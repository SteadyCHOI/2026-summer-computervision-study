"""
Week 2 - Day 4 : 다양한 조건에서의 강건성(robustness) 테스트
AI허브 주행 프레임을 밝은낮/그림자/어두움/흐린차선 등 조건별로 나누어,
고정 파라미터가 어디서 무너지는지 관찰한다.
또한 엣지 검출의 한계를 HSV 색 기반 차선 추출로 보완하고,
둘을 결합해 강건성을 높이는 방법을 실험한다. (4주차 고도화 준비)

실행 방법:
    python week2_day4_robustness.py

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
BLUR_KSIZE = 9

# 2주차에서 찾은 고정 Canny 값 (이게 모든 조건에서 버티는지 테스트)
BEST_LOW = 50
BEST_HIGH = 100

# HSV 색 범위 (영상에 맞게 조정 — inRange 하한/상한)
WHITE_LOW,  WHITE_HIGH  = (0, 0, 160),   (180, 30, 255)     # 흰색 차선
YELLOW_LOW, YELLOW_HIGH = (15, 80, 100), (35, 255, 255)     # 노란색 차선

OUTPUT_DIR = "results_week2_day4"


def load_frame_paths():
    """폴더에서 프레임 이미지 경로를 정렬해 가져온다."""
    paths = sorted(glob.glob(os.path.join(FRAME_DIR, FRAME_EXT)))
    print(f"[load] 찾은 이미지 수: {len(paths)}")
    if len(paths) == 0:
        print("[load] 경고: 이미지를 못 찾음. FRAME_DIR / FRAME_EXT 를 확인하세요.")
    return paths


# ─────────────────────────────────────────────
# 방법 A : 순수 Canny 엣지
# ─────────────────────────────────────────────
def method_canny(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    edges = cv2.Canny(blur, BEST_LOW, BEST_HIGH)
    return edges


# ─────────────────────────────────────────────
# 방법 B : HSV 색 기반 차선 추출
#          흰색/노란색 범위를 골라 차선 색만 남긴다.
#          밝기 변화(그림자)에 덜 휘둘리는 것이 장점.
# ─────────────────────────────────────────────
def method_hsv_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white  = cv2.inRange(hsv, WHITE_LOW,  WHITE_HIGH)    # 흰색만 255
    yellow = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)   # 노란색만 255
    lane_color = cv2.bitwise_or(white, yellow)           # 흰색 OR 노란색
    return lane_color


# ─────────────────────────────────────────────
# 방법 C : 엣지 + 색 결합
#          "차선 색이면서 엣지인 것"만 남겨 강건성을 높인다.
# ─────────────────────────────────────────────
def method_combined(frame):
    edges = method_canny(frame)
    color = method_hsv_color(frame)
    # 색 마스크를 살짝 팽창시켜 엣지와 겹칠 여지를 넓힘
    color_dilated = cv2.dilate(color, np.ones((5, 5), np.uint8), iterations=1)
    combined = cv2.bitwise_and(edges, color_dilated)
    return combined


# ─────────────────────────────────────────────
# 단계 1 : 세 방법을 한 프레임에 나란히 비교
# ─────────────────────────────────────────────
def step1_compare_methods(paths, index=0):
    print(f"\n[단계1] 프레임 {index} 에 세 방법 비교 (Canny / HSV색 / 결합)")
    frame = cv2.imread(paths[index])

    canny    = method_canny(frame)
    color    = method_hsv_color(frame)
    combined = method_combined(frame)

    cv2.imshow("original", frame)
    cv2.imshow("A. canny", canny)
    cv2.imshow("B. hsv color", color)
    cv2.imshow("C. combined", combined)
    print("        아무 키나 누르면 다음 단계로.")
    print("        관찰: 그림자/야간에서 Canny는 흔들려도 색 기반은 버티는지 확인.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# 단계 2 : 여러 조건 프레임을 훑으며 강건성 관찰
#          조건이 다른 프레임(밝은낮/그림자/어두움 등)을 골라
#          같은 방법이 어디서 무너지는지 눈으로 확인.
# ─────────────────────────────────────────────
def step2_scan_conditions(paths, indices=None, delay_ms=1500):
    print("\n[단계2] 여러 조건 프레임 훑기 (q 로 중단)")
    if indices is None:
        # 데이터셋 전체에서 고르게 몇 장 샘플링
        step = max(1, len(paths) // 8)
        indices = list(range(0, len(paths), step))[:8]

    for idx in indices:
        frame = cv2.imread(paths[idx])
        canny    = method_canny(frame)
        combined = method_combined(frame)

        # 원본 / Canny / 결합 을 가로로 붙여 한눈에 비교
        canny_bgr    = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        view = cv2.hconcat([frame, canny_bgr, combined_bgr])
        view = cv2.resize(view, (0, 0), fx=0.5, fy=0.5)  # 화면에 맞게 축소

        cv2.putText(view, f"frame {idx}  (orig | canny | combined)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("condition scan", view)
        if cv2.waitKey(delay_ms) == ord('q'):
            break
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# 단계 3 : HSV 색 범위 실시간 튜닝 (트랙바)
#          흰색/노란색 범위를 슬라이더로 조절해 최적값을 찾는다.
# ─────────────────────────────────────────────
def step3_hsv_trackbar(paths, index=0):
    print("\n[단계3] HSV 흰색 범위 트랙바 튜닝 (q 로 종료)")
    frame = cv2.imread(paths[index])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def nothing(x):
        pass

    cv2.namedWindow("hsv tune")
    # 흰색은 채도(S) 낮고 명도(V) 높음 → S 상한, V 하한을 조절해보는 게 핵심
    cv2.createTrackbar("S_max", "hsv tune", 30,  255, nothing)
    cv2.createTrackbar("V_min", "hsv tune", 200, 255, nothing)

    while True:
        s_max = cv2.getTrackbarPos("S_max", "hsv tune")
        v_min = cv2.getTrackbarPos("V_min", "hsv tune")
        white = cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))
        cv2.imshow("hsv tune", white)
        if cv2.waitKey(1) == ord('q'):
            print(f"        선택값 → S_max={s_max}, V_min={v_min}")
            print("        이 값을 WHITE_LOW/WHITE_HIGH 에 반영하세요.")
            break
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# 단계 4 : 조건별 대표 프레임 결과 저장 (보고서용)
# ─────────────────────────────────────────────
def step4_save(paths, indices=None, ):
    print(f"\n[단계4] 대표 프레임 결과 저장 → '{OUTPUT_DIR}/'")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if indices is None:
        step = max(1, len(paths) // 4)
        indices = list(range(0, len(paths), step))[:4]

    for idx in indices:
        frame = cv2.imread(paths[idx])
        canny    = cv2.cvtColor(method_canny(frame),    cv2.COLOR_GRAY2BGR)
        color    = cv2.cvtColor(method_hsv_color(frame), cv2.COLOR_GRAY2BGR)
        combined = cv2.cvtColor(method_combined(frame), cv2.COLOR_GRAY2BGR)

        # 원본 | Canny | 색 | 결합  네 개를 가로로
        view = cv2.hconcat([frame, canny, color, combined])
        out_path = os.path.join(OUTPUT_DIR, f"robust_{idx}.jpg")
        cv2.imwrite(out_path, view)
        print(f"        저장: {out_path}  (원본|canny|color|combined)")


# ─────────────────────────────────────────────
# 단계 5 : 강건성 분석 정리 (콘솔 출력)
# ─────────────────────────────────────────────
def step5_print_analysis():
    print("\n[단계5] 조건별 강건성 분석 (관찰 결과를 채워 넣으세요)")
    print("  ┌──────────────┬─────────────┬─────────────┬─────────────┐")
    print("  │ 조건         │ Canny       │ HSV 색      │ 결합         │")
    print("  ├──────────────┼─────────────┼─────────────┼─────────────┤")
    print("  │ 밝은 낮      │ 양호        │ 양호        │ 양호        │")
    print("  │ 그림자       │ 약함(가짜선)│ 강함        │ 강함        │")
    print("  │ 야간/터널    │ 약함(소실)  │ 중간        │ 중간        │")
    print("  │ 흐린 차선    │ 약함        │ 조건부      │ 약함        │")
    print("  └──────────────┴─────────────┴─────────────┴─────────────┘")
    print("  → 4주차 개선 방향: 그림자에는 색 기반 보완, 야간에는")
    print("     대비 향상(히스토그램 평활화) 또는 적응형 임계값 검토.")


# ─────────────────────────────────────────────
# main : 단계별 순차 실행
#        필요 없는 단계는 주석 처리하면 됨
# ─────────────────────────────────────────────
if __name__ == "__main__":
    paths = load_frame_paths()

    if len(paths) > 0:
        step1_compare_methods(paths, index=0)   # 세 방법 비교
        step2_scan_conditions(paths)            # 여러 조건 훑기
        step3_hsv_trackbar(paths, index=0)      # HSV 범위 튜닝
        step4_save(paths)                       # 결과 저장
        step5_print_analysis()                  # 분석 정리

    print("\n[done] 종료")