"""
Week 2 - Day 5 : 통합 전처리 파이프라인 + 프레임 시퀀스 적용
Day 1~4에서 만든 조각(Canny, HSV 색, 결합, ROI)을 하나의 파이프라인으로 통합한다.
프레임 시퀀스 전체에 실시간 적용해 안정성을 확인하고,
3주차 Hough Transform 입력이 될 깨끗한 엣지를 출력한다.
처리 결과를 영상(mp4)으로도 저장할 수 있다.

실행 방법:
    python week2_day5_integrated.py

각 단계는 함수로 분리되어 있고, main에서 순서대로 호출한다.
"""

import cv2
import glob
import os
import numpy as np

# ─────────────────────────────────────────────
# 설정 (여기만 네 환경에 맞게 수정)
# ─────────────────────────────────────────────
FRAME_DIR = "도로주행영상/Validation/원천데이터/bb/자동차전용도로/주간일출/맑음/30_전방/20200910_104924/1/"
FRAME_EXT = "*.jpg"
BLUR_KSIZE = 9

# Day2 에서 찾은 Canny 값
BEST_LOW = 50
BEST_HIGH = 100

# Day4 에서 찾은 HSV 색 범위 (트랙바 결과 반영)
WHITE_LOW,  WHITE_HIGH  = (0, 0, 160),   (180, 30, 255)
YELLOW_LOW, YELLOW_HIGH = (15, 80, 100), (35, 255, 255)

# 사다리꼴 ROI 좌표 비율 (영상 도로 위치에 맞게 조정)
ROI_POLYGON_RATIO = [
    (0.00, 1.00),   # 좌하단
    (0.35, 0.00),   # 좌상단
    (0.65, 0.00),   # 우상단
    (1.00, 1.00),   # 우하단
]

OUTPUT_DIR = "results_week2_day5"


def load_frame_paths():
    paths = sorted(glob.glob(os.path.join(FRAME_DIR, FRAME_EXT)))
    print(f"[load] 찾은 이미지 수: {len(paths)}")
    if len(paths) == 0:
        print("[load] 경고: 이미지를 못 찾음. FRAME_DIR / FRAME_EXT 를 확인하세요.")
    return paths


# ─────────────────────────────────────────────
# 조각 함수들 (Day 1~4 에서 만든 것)
# ─────────────────────────────────────────────
def get_color_mask(frame):
    """HSV 흰색+노란색 차선 마스크"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white  = cv2.inRange(hsv, WHITE_LOW,  WHITE_HIGH)
    yellow = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    return cv2.bitwise_or(white, yellow)


def apply_trapezoid_roi(edges):
    """사다리꼴 ROI 마스킹 (Canny 이후에 적용해 가짜 엣지 방지)"""
    h, w = edges.shape[:2]
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(w * rx), int(h * ry)) for (rx, ry) in ROI_POLYGON_RATIO
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


# ─────────────────────────────────────────────
# 단계 1 : 통합 전처리 파이프라인 (2주차 최종 산출물)
#          플래그로 색 결합/ROI 를 켜고 끌 수 있다.
# ─────────────────────────────────────────────
def preprocess(frame, use_color=True, use_roi=True):
    """
    프레임 → 깨끗한 차선 엣지 (3주차 Hough 입력용)
    처리 순서:
      1) 직사각형 크롭 (연산 절감 — 하늘/건물 미리 제거)
      2) gray → blur → Canny
      3) HSV 색 마스크와 결합 (선택)
      4) 사다리꼴 ROI (선택, 가짜엣지 방지 위해 마지막)
    반환: (원본크기의) 엣지 이미지
    """
    h, w = frame.shape[:2]

    # 1) 직사각형 크롭: 하단 절반만 실제로 잘라 연산량 절감
    top = h // 2
    cropped = frame[top:h, 0:w]

    # 2) gray → blur → Canny
    gray  = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    edges = cv2.Canny(blur, BEST_LOW, BEST_HIGH)

    # 3) 색 결합 (선택): 차선 색이면서 엣지인 것만
    if use_color:
        color = get_color_mask(cropped)
        color = cv2.dilate(color, np.ones((5, 5), np.uint8), iterations=1)
        edges = cv2.bitwise_and(edges, color)

    # 4) 사다리꼴 ROI (선택): 크롭된 이미지 기준으로 적용
    if use_roi:
        edges = apply_trapezoid_roi(edges)

    # 원본 크기 캔버스에 다시 얹어 반환 (좌표 일관성 유지)
    full = np.zeros((h, w), np.uint8)
    full[top:h, 0:w] = edges
    return full


# ─────────────────────────────────────────────
# 단계 2 : 프레임 시퀀스 실시간 재생
#          영상처럼 프레임을 넘기며 파이프라인 동작 확인.
# ─────────────────────────────────────────────
def step2_play_sequence(paths, delay_ms=30):
    print("\n[단계2] 프레임 시퀀스 재생 (q 로 종료)")
    print("        원본 | 최종 엣지 를 나란히 표시")
    for path in paths:
        frame = cv2.imread(path)
        edges = preprocess(frame)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        view = cv2.hconcat([frame, edges_bgr])
        view = cv2.resize(view, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("original | pipeline", view)

        if cv2.waitKey(delay_ms) == ord('q'):
            break
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# 단계 3 : 옵션 비교 (색 결합 유무, ROI 유무)
#          어떤 조합이 가장 깨끗한지 한 프레임에서 확인.
# ─────────────────────────────────────────────
def step3_compare_options(paths, index=0):
    print(f"\n[단계3] 파이프라인 옵션 비교 (프레임 {index})")
    frame = cv2.imread(paths[index])

    plain    = preprocess(frame, use_color=False, use_roi=False)
    roi_only = preprocess(frame, use_color=False, use_roi=True)
    full     = preprocess(frame, use_color=True,  use_roi=True)

    cv2.imshow("original", frame)
    cv2.imshow("A. canny only", plain)
    cv2.imshow("B. canny + roi", roi_only)
    cv2.imshow("C. canny + color + roi (final)", full)
    print("        아무 키나 누르면 다음 단계로.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# 단계 4 : 처리 결과를 영상(mp4)으로 저장
#          before/after 데모 영상 제작 (발표/블로그용)
# ─────────────────────────────────────────────
def step4_export_video(paths, fps=20, max_frames=200):
    print(f"\n[단계4] 결과 영상 저장 → '{OUTPUT_DIR}/output.mp4'")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(paths) == 0:
        return

    # 첫 프레임으로 출력 크기 결정 (원본+엣지 가로 결합, 0.5배)
    sample = cv2.imread(paths[0])
    h, w = sample.shape[:2]
    out_w, out_h = w, h // 2 * 2   # 짝수 보정
    # 가로 결합(원본|엣지) 후 0.5배 → 최종 크기
    combined_w = int(w * 2 * 0.5)
    combined_h = int(h * 0.5)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(OUTPUT_DIR, "output.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (combined_w, combined_h))

    for path in paths[:max_frames]:
        frame = cv2.imread(path)
        edges = preprocess(frame)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        view = cv2.hconcat([frame, edges_bgr])
        view = cv2.resize(view, (combined_w, combined_h))
        writer.write(view)

    writer.release()
    print(f"        저장 완료: {out_path}  ({min(len(paths), max_frames)} 프레임)")


# ─────────────────────────────────────────────
# 단계 5 : 대표 프레임 저장 + 2주차 회고
# ─────────────────────────────────────────────
def step5_save_and_summary(paths, save_n=3):
    print(f"\n[단계5] 대표 프레임 {save_n}장 저장 + 회고")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    step = max(1, len(paths) // save_n)
    for i in range(save_n):
        idx = i * step
        frame = cv2.imread(paths[idx])
        edges = cv2.cvtColor(preprocess(frame), cv2.COLOR_GRAY2BGR)
        view = cv2.hconcat([frame, edges])
        out_path = os.path.join(OUTPUT_DIR, f"final_{idx}.jpg")
        cv2.imwrite(out_path, view)
        print(f"        저장: {out_path}")

    print("\n  === 2주차 회고 ===")
    print("  - Day1: 영상 대신 AI허브 프레임 시퀀스로 데이터 확보")
    print("  - Day2: 트랙바로 Canny 파라미터 실시간 튜닝")
    print("  - Day3: 엣지 기법 비교(Canny 채택) + 사다리꼴 ROI")
    print("  - Day4: 조건별 강건성 테스트 + HSV 색 보완")
    print("  - Day5: 전체 통합 파이프라인 완성 → 3주차 Hough 입력 준비 완료")


# ─────────────────────────────────────────────
# main : 단계별 순차 실행
#        필요 없는 단계는 주석 처리하면 됨
# ─────────────────────────────────────────────
if __name__ == "__main__":
    paths = load_frame_paths()

    if len(paths) > 0:
        step3_compare_options(paths, index=0)   # 옵션 비교 (먼저 최적 조합 확인)
        step2_play_sequence(paths)              # 시퀀스 재생
        step4_export_video(paths)               # 결과 영상 저장
        step5_save_and_summary(paths)           # 대표 저장 + 회고

    print("\n[done] 종료")