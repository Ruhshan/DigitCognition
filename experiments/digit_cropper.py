

import cv2
from PIL import Image
image = cv2.imread("roi.png")
import uuid

base = Image.open("data/base.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 3))
#dilated = cv2.dilate(thresh,kernel,iterations = 1)
dilated = cv2.dilate(src=thresh, kernel=kernel, anchor=(-1, -1), iterations=2)
_, contours, hierarchy = cv2.findContours(
	dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("what", dilated)

i = 5

padding = 2
for contour in contours:

	[x,y,w,h] = cv2.boundingRect(contour)
	area = w*h
	print(area)
	if area>100 and area < 2000:
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
		#cv2.imwrite(str(i)+".jpg",image[y:y+h,x:x+w])
		single = gray[y-padding:y+h+padding,x-padding:x+w+padding]
		ret,thresh = cv2.threshold(single,127,255,cv2.THRESH_BINARY)
		#cv2.imshow("crp"+str(i), thresh1)
		#threshr = cv2.resize(thresh, (25, 50))
		imarr = Image.fromarray(thresh)
		#cv2.imwrite("data/zero/"+str(i)+".png", thresh)
		base.paste(imarr)
		base.save("data/one/"+str(uuid.uuid1())+".png", "PNG")
		i=i+1
# cv2.imshow("cont", image)
cv2.imshow("rect", image)

cv2.waitKey(0)
