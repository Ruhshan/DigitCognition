import cv2
read = cv2.imread('square_frame.png',1)
cv2.imshow('f',read)

cv2.waitKey(0)
cv2.destroyAllWindows()
