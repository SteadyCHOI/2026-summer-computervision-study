import cv2

img = cv2.imread("test.jpg")
print(img)               # (높이, 너비, 3) 이 찍히면 성공
cv2.imshow("test", img) # 창 띄우는 코드
cv2.waitKey(0) # 키 입력할때까지 안꺼지고 계속 기다림
cv2.destroyAllWindows() # 지워질때 창 전부 삭제