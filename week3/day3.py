"""
Week 3 - Day 3 : 선분 통합 + 차선 영역 시각화 (차선 인식 1차 완성)
Day1(Hough 검출) → Day2(각도 필터 + 좌/우 분류) → Day3(선분 평균 + 영역 채우기)
여러 짧은 선분을 좌/우 각각 하나의 대표 차선으로 통합하고,
두 차선 사이를 반투명으로 채워 시각화한다.

실행 방법:
    python week3_day3_lane_pipeline.py

트랙바:
    threshold / minLen / maxGap : Hough 파라미터
    slopeMin(x100)             : 이 기울기보다 완만하면 버림 (50 = 0.5)
"""

import cv2
import numpy as np
from week2.day5 import preprocess

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
IMG_PATH = "도로주행영상/Validation/원천데이터/bb/자동차전용도로/주간일출/맑음/30_전방/20200910_104924/1/1_20200910_104924_000000.jpg"
WINDOW_NAME = "lane pipeline"

Y_TOP_RATIO = 0.60      # 차선을 그릴 위쪽 y 위치 (화면 높이의 60% 지점)


def nothing(x):
    pass


# ─────────────────────────────────────────────
# Day2 : 각도 필터 + 좌/우 분류
# ─────────────────────────────────────────────
def classify_lines(lines, frame_width, slope_min):
    left_lines, right_lines = [], []
    mid = frame_width // 2
    if lines is None:
        return left_lines, right_lines

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:                 # 수직 → 0으로 나누기 방지
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < slope_min:       # 수평 잡선 제거
            continue
        if slope < 0 and x1 < mid and x2 < mid:
            left_lines.append(line)
        elif slope > 0 and x1 > mid and x2 > mid:
            right_lines.append(line)
    return left_lines, right_lines


# ─────────────────────────────────────────────
# Day3 : 선분들을 하나의 대표 직선(기울기, 절편)으로 평균
#        길이를 가중치로 사용해 긴 선분에 더 큰 비중을 준다.
# ─────────────────────────────────────────────
def average_lane(lines):
    slopes, intercepts, weights = [], [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # 선분 길이 = 가중치
        slopes.append(slope)
        intercepts.append(intercept)
        weights.append(length)

    if len(slopes) == 0:
        return None
    avg_slope = np.average(slopes, weights=weights)
    avg_intercept = np.average(intercepts, weights=weights)
    return avg_slope, avg_intercept


# ─────────────────────────────────────────────
# Day3 : (기울기, 절편) → 그릴 수 있는 끝점 좌표로 변환
#        y_bottom~y_top 구간에 해당하는 x 를 역산.
# ─────────────────────────────────────────────
def make_line_points(y_bottom, y_top, lane):
    if lane is None:
        return None
    slope, intercept = lane
    if slope == 0:
        return None
    x1 = int((y_bottom - intercept) / slope)   # y = ax + b → x = (y-b)/a
    x2 = int((y_top - intercept) / slope)
    return (x1, y_bottom, x2, y_top)


# ─────────────────────────────────────────────
# Day3 : 대표 차선 2개 그리기 + 사이 영역 반투명 채우기
# ─────────────────────────────────────────────
def draw_lane(frame, left_pts, right_pts):
    result = frame.copy()
    overlay = frame.copy()

    if left_pts:
        cv2.line(result, (left_pts[0], left_pts[1]),
                 (left_pts[2], left_pts[3]), (0, 255, 0), 6)
    if right_pts:
        cv2.line(result, (right_pts[0], right_pts[1]),
                 (right_pts[2], right_pts[3]), (0, 255, 0), 6)

    # 두 차선이 모두 있으면 사이를 채움
    if left_pts and right_pts:
        pts = np.array([[
            (left_pts[0],  left_pts[1]),    # 좌 아래
            (left_pts[2],  left_pts[3]),    # 좌 위
            (right_pts[2], right_pts[3]),   # 우 위
            (right_pts[0], right_pts[1]),   # 우 아래
        ]], np.int32)
        cv2.fillPoly(overlay, pts, (0, 255, 0))
        result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)  # 반투명 합성

    return result


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    frame = cv2.imread(IMG_PATH)
    if frame is None:
        print("이미지를 못 읽음. IMG_PATH 확인.")
        return

    h = frame.shape[0]
    y_bottom = h
    y_top = int(h * Y_TOP_RATIO)

    edges = preprocess(frame, use_color=False, use_roi=False)

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar("threshold", WINDOW_NAME, 30,  300,  nothing)
    cv2.createTrackbar("minLen",    WINDOW_NAME, 40,  300,  nothing)
    cv2.createTrackbar("maxGap",    WINDOW_NAME, 150, 1000, nothing)
    cv2.createTrackbar("slopeMin",  WINDOW_NAME, 50,  200,  nothing)  # x100

    while True:
        threshold = max(1, cv2.getTrackbarPos("threshold", WINDOW_NAME))
        min_len   = cv2.getTrackbarPos("minLen",    WINDOW_NAME)
        max_gap   = cv2.getTrackbarPos("maxGap",    WINDOW_NAME)
        slope_min = cv2.getTrackbarPos("slopeMin",  WINDOW_NAME) / 100.0

        # Day1: Hough 검출
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                threshold=threshold,
                                minLineLength=min_len,
                                maxLineGap=max_gap)

        # Day2: 각도 필터 + 좌/우 분류
        left_lines, right_lines = classify_lines(lines, frame.shape[1], slope_min)

        # Day3: 평균 → 좌표 변환 → 그리기
        left_lane  = average_lane(left_lines)
        right_lane = average_lane(right_lines)
        left_pts   = make_line_points(y_bottom, y_top, left_lane)
        right_pts  = make_line_points(y_bottom, y_top, right_lane)
        result = draw_lane(frame, left_pts, right_pts)

        # 상태 표시
        l_ok = "O" if left_pts else "X"
        r_ok = "O" if right_pts else "X"
        cv2.putText(result, f"Left:{l_ok}  Right:{r_ok}  slopeMin={slope_min:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, result)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord("q"):
            print(f"최종값 → threshold={threshold}, minLen={min_len}, "
                  f"maxGap={max_gap}, slopeMin={slope_min:.2f}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()