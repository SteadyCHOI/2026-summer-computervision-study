import cv2

img = cv2.imread("test.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 엣지는 보통 흑백으로

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # 가로방향 미분
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # 세로방향 미분

#cv2.imshow("sobel_x", sobel_x)
#cv2.imshow("sobel_y", sobel_y)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

edges1 = cv2.Canny(gray, 100, 200)
edges2 = cv2.Canny(gray, 50, 100)
edges3 = cv2.Canny(gray, 200, 300)

cv2.imshow("canny medium", edges1)
cv2.imshow("canny low", edges2)
cv2.imshow("canny high", edges3)
cv2.waitKey(0)
cv2.destroyAllWindows()
