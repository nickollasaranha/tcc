import cv2
import numpy as np
import math

def auto_canny(image, sigma=0.33):
  # compute the median of the single channel pixel intensities
  v = np.median(image)

  # apply automatic Canny edge detection using the computed median
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)
  return edged

# this returns in degrees
def horizontal_angle(line):
    return 0 if line[2]-line[0] == 0 else abs(math.degrees(math.atan(((line[3]-line[1])/(line[2]-line[0])))))

img = cv2.imread('track2.png', 0)

canny = auto_canny(img.copy())
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, 1)
lines = lsd.detect(img)
drawn_img = lsd.drawSegments(np.zeros((1080, 1920, 3), np.uint8), lines[0])


cv2.imshow('image3', drawn_img)
cv2.waitKey(0)
cv2.destroyAllWindows()