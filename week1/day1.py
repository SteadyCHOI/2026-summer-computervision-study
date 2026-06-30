import cv2

img = cv2.imread("test.jpg")
print(img)
print(img.shape)
print(img.dtype)

img[0:50, 0:50] = [0, 0, 255]

cv2.imshow("test", img)
cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print("원본 :", img.shape)    # (480, 640, 3)
print("흑백 :", gray.shape)   # (480, 640)      ← 채널이 사라짐!
print("HSV  :", hsv.shape)    # (480, 640, 3)