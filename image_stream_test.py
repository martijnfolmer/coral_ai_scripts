import cv2


img = cv2.imread('test.png')
cv2.imshow('frame', img)
cv2.waitKey(-1)
