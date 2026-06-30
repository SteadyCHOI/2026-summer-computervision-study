import cv2
import numpy as np

img = cv2.imread("test.jpg")

kernel = np.ones((3, 3), np.float32) / 9
custom = cv2.filter2D(img, -1, kernel)

blur1 = cv2.GaussianBlur(img, (5, 5), 0)
blur2 = cv2.GaussianBlur(img, (15, 15), 0)
blur3 = cv2.GaussianBlur(img, (25, 25), 0)
m_blur = cv2.medianBlur(img, 5)

cv2.imshow("gaussian1", blur1)
cv2.imshow("gaussian2", blur2)
cv2.imshow("gaussian3", blur3)
cv2.imshow("median", m_blur)
cv2.imshow("custom", custom)
cv2.waitKey(0)
cv2.destroyAllWindows()