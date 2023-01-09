import cv2

imgPath = 'FT_unitTest/unitTest (1).png'
img = cv2.imread(imgPath)
cv2.imshow("this window", img)
cv2.waitKey(-1)
