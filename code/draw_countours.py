import cv2 
import numpy as np

def auto_canny(image, sigma=0.33):
  # compute the median of the single channel pixel intensities
  v = np.median(image)

  # apply automatic Canny edge detection using the computed median
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)

  # return the edged image
  return edged

image = cv2.imread("track1.png", 0)
equalized_image = cv2.equalizeHist(image)
canny_imaged = auto_canny(equalized_image)
dilated_image = cv2.dilate(canny_imaged, np.ones((1, 5), np.uint8), iterations = 1)

#ret, thresh = cv2.threshold(darker, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#newimg = cv2.bitwise_not(thresh)

im2, contours, hierarchy = cv2.findContours(dilated_image.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
	cv2.drawContours(im2, [cnt], 0, 255, -1)

cv2.imshow("edit1", dilated_image)
cv2.imshow("edit", im2)
cv2.waitKey(0)
cv2.destroyAllWindows()