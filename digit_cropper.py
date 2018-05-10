

import cv2

image = cv2.imread("croped0.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 3))
#dilated = cv2.dilate(thresh,kernel,iterations = 1)
dilated = cv2.dilate(src=thresh, kernel=kernel, anchor=(-1, -1), iterations=2)
_, contours, hierarchy = cv2.findContours(
	dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#cv2.imshow("what", dilated)

i = 5

padding = 2
for contour in contours:

	[x,y,w,h] = cv2.boundingRect(contour)
	area = w*h
	if area>50 and area < 500:
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

		#cv2.imwrite(str(i)+".jpg",image[y:y+h,x:x+w])
		single = gray[y-padding:y+h+padding,x-padding:x+w+padding]
		ret,thresh1 = cv2.threshold(single,127,255,cv2.THRESH_BINARY)
		cv2.imshow("crp"+str(i), thresh1)
		i=i+1
# cv2.imshow("cont", image)
cv2.imshow("rect", image)

cv2.waitKey(0)
