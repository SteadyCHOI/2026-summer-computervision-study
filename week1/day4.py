import cv2
import numpy as np

img = cv2.imread("test.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

kernel = np.ones((3, 3), np.uint8)   # 모폴로지용 커널

dilated = cv2.dilate(edges, kernel, iterations=2)   # 팽창: 흰 영역 불림
eroded  = cv2.erode(edges, kernel, iterations=0)    # 침식: 흰 영역 깎음

cv2.imshow("edges", edges)
cv2.imshow("dilated", dilated)
cv2.imshow("eroded", eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(contours)

# 원본에 외곽선 그려보기
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)   # 초록색으로

cv2.imshow("contours", result)
cv2.waitKey(0)
cv2.destroyAllWindows()