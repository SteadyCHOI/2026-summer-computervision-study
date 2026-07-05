import cv2
import glob

# 폴더 안 모든 jpg 경로를 정렬해서 가져오기
frame_paths = sorted(glob.glob("도로주행영상/Validation/원천데이터/bb/자동차전용도로/주간일출/맑음/30_전방/20200910_104924/1/*.jpg"))

for path in frame_paths:
    frame = cv2.imread(path)      # 1주차의 그 imread 그대로!
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 100)
    cv2.imshow("edges", edges)
    if cv2.waitKey(150) == ord('q'):   # 프레임을 25ms 간격으로 넘겨 영상처럼 재생
        break

cv2.destroyAllWindows()