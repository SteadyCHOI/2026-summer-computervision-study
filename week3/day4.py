"""
Week 3 - Day 4 : 동영상(프레임 시퀀스) 차선 인식
Day1~3 의 단일 프레임 파이프라인을 process_frame() 으로 묶고,
프레임 시퀀스 전체에 적용해 실시간 재생한다.
- 차선을 못 잡은 프레임은 이전 값을 유지해 깜빡임(jitter) 완화
- 처리 결과를 mp4 로 저장 (발표/블로그용 데모)

실행 방법:
    python week3_day4_video.py
    q 또는 ESC 로 종료

주의: FRAME_DIR 는 '연속된 프레임'이 든 폴더여야 영상처럼 재생됨.
"""

import cv2
import glob
import os
import numpy as np
from week2.day5 import preprocess

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
FRAME_DIR = "도로주행영상/Validation/원천데이터/bb/자동차전용도로/주간일출/맑음/30_전방/20200910_104924/1"
FRAME_EXT = "*.jpg"

# Day1~3 에서 찾은 파라미터
PARAMS = {
    "threshold": 30,
    "minLen": 40,
    "maxGap": 150,
    "slopeMin": 0.5,
}

Y_TOP_RATIO = 0.60
PLAY_DELAY_MS = 30          # 프레임 간격 (30≈33fps, 크게 하면 슬로우)
SAVE_VIDEO = True           # 결과 영상 저장 여부
OUTPUT_DIR = "results_week3"
OUTPUT_FPS = 20


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
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < slope_min:
            continue
        if slope < 0 and x1 < mid and x2 < mid:
            left_lines.append(line)
        elif slope > 0 and x1 > mid and x2 > mid:
            right_lines.append(line)
    return left_lines, right_lines


# ─────────────────────────────────────────────
# Day3 : 선분 평균 (길이 가중)
# ─────────────────────────────────────────────
def average_lane(lines):
    slopes, intercepts, weights = [], [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        slopes.append(slope)
        intercepts.append(intercept)
        weights.append(length)
    if len(slopes) == 0:
        return None
    return np.average(slopes, weights=weights), np.average(intercepts, weights=weights)


# ─────────────────────────────────────────────
# Day3 : (기울기,절편) → 끝점 좌표
# ─────────────────────────────────────────────
def make_line_points(y_bottom, y_top, lane):
    if lane is None:
        return None
    slope, intercept = lane
    if slope == 0:
        return None
    x1 = int((y_bottom - intercept) / slope)
    x2 = int((y_top - intercept) / slope)
    return (x1, y_bottom, x2, y_top)


# ─────────────────────────────────────────────
# Day3 : 차선 그리기 + 영역 채우기
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
    if left_pts and right_pts:
        pts = np.array([[
            (left_pts[0], left_pts[1]), (left_pts[2], left_pts[3]),
            (right_pts[2], right_pts[3]), (right_pts[0], right_pts[1]),
        ]], np.int32)
        cv2.fillPoly(overlay, pts, (0, 255, 0))
        result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)
    return result


# ─────────────────────────────────────────────
# Day4 : 프레임 1장 → 차선 끝점 검출 (그리기 전 단계까지)
#        이전값 유지를 위해 끝점만 반환하도록 분리.
# ─────────────────────────────────────────────
def detect_lane_points(frame, params):
    h = frame.shape[0]
    y_bottom, y_top = h, int(h * Y_TOP_RATIO)

    edges = preprocess(frame, use_color=False, use_roi=False)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            params["threshold"], params["minLen"], params["maxGap"])
    left_lines, right_lines = classify_lines(lines, frame.shape[1], params["slopeMin"])
    left_pts = make_line_points(y_bottom, y_top, average_lane(left_lines))
    right_pts = make_line_points(y_bottom, y_top, average_lane(right_lines))
    return left_pts, right_pts


# ─────────────────────────────────────────────
# main : 시퀀스 재생 + 이전값 유지 + 영상 저장
# ─────────────────────────────────────────────
def main():
    frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR, FRAME_EXT)))
    print(f"프레임 수: {len(frame_paths)}")
    if len(frame_paths) == 0:
        print("프레임을 못 찾음. FRAME_DIR 확인.")
        return

    # 영상 저장 준비
    writer = None
    if SAVE_VIDEO:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        h, w = cv2.imread(frame_paths[0]).shape[:2]
        writer = cv2.VideoWriter(
            os.path.join(OUTPUT_DIR, "lane_output.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, (w, h))

    prev_left, prev_right = None, None   # 이전 프레임 차선 (구멍 메우기용)

    for path in frame_paths:
        frame = cv2.imread(path)
        left_pts, right_pts = detect_lane_points(frame, PARAMS)

        # 못 잡은 쪽은 이전 값으로 유지 → 깜빡임 완화
        if left_pts is None:
            left_pts = prev_left
        if right_pts is None:
            right_pts = prev_right
        prev_left, prev_right = left_pts, right_pts

        result = draw_lane(frame, left_pts, right_pts)

        l_ok = "O" if left_pts else "X"
        r_ok = "O" if right_pts else "X"
        cv2.putText(result, f"Left:{l_ok}  Right:{r_ok}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if writer is not None:
            writer.write(result)

        cv2.imshow("lane detection (video)", result)
        key = cv2.waitKey(PLAY_DELAY_MS) & 0xFF
        if key == 27 or key == ord("q"):
            break

    if writer is not None:
        writer.release()
        print(f"영상 저장 완료 → {OUTPUT_DIR}/lane_output.mp4")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()