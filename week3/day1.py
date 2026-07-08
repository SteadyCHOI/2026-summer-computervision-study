import cv2
import numpy as np
from week2.day5 import preprocess

def nothing(x):
    pass

frame = cv2.imread("도로주행영상/Validation/원천데이터/bb/자동차전용도로/주간일출/맑음/30_전방/20200910_104924/1/1_20200910_104924_000000.jpg")
edges = preprocess(frame, use_roi=False)     # 2주차 통합 함수 (깨끗한 엣지)

WINDOW_NAME = "hough lines"
cv2.namedWindow(WINDOW_NAME)
cv2.createTrackbar("threshold", WINDOW_NAME, 10, 300, nothing)
cv2.createTrackbar("minLen", WINDOW_NAME, 10, 300, nothing)
cv2.createTrackbar("maxGap", WINDOW_NAME, 7, 300, nothing)

while True:
    threshold = max(1, cv2.getTrackbarPos("threshold", WINDOW_NAME))
    min_len = cv2.getTrackbarPos("minLen", WINDOW_NAME)
    max_gap = cv2.getTrackbarPos("maxGap", WINDOW_NAME)

    lines = cv2.HoughLinesP(
        edges,              # 입력: Canny 엣지 이미지
        rho=1,              # ρ 격자 해상도 (1픽셀 단위)
        theta=np.pi/180,    # θ 격자 해상도 (1도 단위, 라디안으로)
        threshold=threshold,       # 최소 투표 수 (이만큼 표 모여야 직선 인정)
        minLineLength=min_len,   # 이보다 짧은 선분은 무시
        maxLineGap=max_gap      # 이 간격 이내로 끊긴 선은 하나로 이음
    )

    result = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imshow(WINDOW_NAME, result)

    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord("q"):  # ESC 또는 q 키로 종료
        break

cv2.destroyAllWindows()