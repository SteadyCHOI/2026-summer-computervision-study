"""
Week 3 - Day 2 : 기울기 필터링 + 좌/우 차선 분류
Day1 의 Hough 검출 결과에서 각도(기울기)로 잡선(가드레일, 가로 엣지 등)을 제거하고,
기울기 부호로 좌/우 차선을 분류한다.
좌=파랑, 우=빨강 으로 시각화해 분류가 잘 됐는지 확인한다.

실행 방법:
    python week3_day2_slope_filter.py

트랙바:
    threshold / minLen / maxGap : Hough 파라미터 (Day1 과 동일)
    slopeMin(x100)             : 이 기울기보다 완만하면 버림 (수평 잡선 제거)
                                 값은 x100 정수라 50 = 기울기 0.5
"""

import cv2
import numpy as np
from week2.day5 import preprocess

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
IMG_PATH = "도로주행영상/Validation/원천데이터/bb/자동차전용도로/주간일출/맑음/30_전방/20200910_104924/1/1_20200910_104924_000000.jpg"
WINDOW_NAME = "lane filter"


def nothing(x):
    pass


def classify_lines(lines, frame_width, slope_min):
    """
    선분들을 각도로 필터링하고 좌/우로 분류한다.
    - 기울기 절댓값이 slope_min 미만이면 수평 잡선으로 보고 버림
    - 이미지 좌표계라 y축이 뒤집혀 있음:
        왼쪽 차선 = 음의 기울기, 오른쪽 차선 = 양의 기울기
    - 위치(화면 좌/우 절반)로도 검증해 견고성을 높임
    반환: (left_lines, right_lines)
    """
    left_lines, right_lines = [], []
    mid = frame_width // 2

    if lines is None:
        return left_lines, right_lines

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:                 # 완전 수직 → 0으로 나누기 방지
            continue
        slope = (y2 - y1) / (x2 - x1)

        if abs(slope) < slope_min:       # 너무 수평 → 차선 아님(가로 엣지/가드레일)
            continue

        # 기울기 부호 + 위치로 좌/우 분류
        if slope < 0 and x1 < mid and x2 < mid:
            left_lines.append(line)
        elif slope > 0 and x1 > mid and x2 > mid:
            right_lines.append(line)

    return left_lines, right_lines


def draw_lines(frame, left_lines, right_lines):
    """좌=파랑, 우=빨강 으로 그린다."""
    result = frame.copy()
    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 3)   # 파랑 = 왼쪽
    for line in right_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)   # 빨강 = 오른쪽
    return result


def main():
    frame = cv2.imread(IMG_PATH)
    if frame is None:
        print("이미지를 못 읽음. IMG_PATH 확인.")
        return

    # 색·ROI 끈 깨끗한 엣지 (Day1 에서 확인된 최적 입력)
    edges = preprocess(frame, use_color=False, use_roi=False)

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar("threshold", WINDOW_NAME, 10,  300,  nothing)
    cv2.createTrackbar("minLen",    WINDOW_NAME, 10,  300,  nothing)
    cv2.createTrackbar("maxGap",    WINDOW_NAME, 7, 1000, nothing)
    cv2.createTrackbar("slopeMin",  WINDOW_NAME, 20,  200,  nothing)  # x100 → 50=0.5

    while True:
        threshold = max(1, cv2.getTrackbarPos("threshold", WINDOW_NAME))
        min_len   = cv2.getTrackbarPos("minLen",    WINDOW_NAME)
        max_gap   = cv2.getTrackbarPos("maxGap",    WINDOW_NAME)
        slope_min = cv2.getTrackbarPos("slopeMin",  WINDOW_NAME) / 100.0

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_len,
            maxLineGap=max_gap,
        )

        left_lines, right_lines = classify_lines(lines, frame.shape[1], slope_min)
        result = draw_lines(frame, left_lines, right_lines)

        # 검출 개수 표시
        cv2.putText(result,
                    f"L(blue)={len(left_lines)}  R(red)={len(right_lines)}  slopeMin={slope_min:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, result)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord("q"):
            print(f"최종값 → threshold={threshold}, minLen={min_len}, "
                  f"maxGap={max_gap}, slopeMin={slope_min:.2f}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()