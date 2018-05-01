import cv2
import numpy as np
import math

def draw(img):

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def auto_canny(image, sigma=0.33):

  # compute the median of the single channel pixel intensities
  v = np.median(image)

  # apply automatic Canny edge detection using the computed median
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)

  # return the edged image
  return edged

img = cv2.imread('track1.png', 0)
img = auto_canny(img)

def fill_holes(image):

  # Copy the thresholded image.
  im_floodfill = image.copy()

  # Mask used to flood filling.
  # Notice the size needs to be 2 pixels than the image.
  h, w = image.shape[:2]
  mask = np.zeros((h+2, w+2), np.uint8)
   
  # Floodfill from point (0, 0)
  cv2.floodFill(im_floodfill, mask, (0,0), 255);
   
  # Invert floodfilled image
  im_floodfill_inv = cv2.bitwise_not(im_floodfill)
   
  # Combine the two images to get the foreground.
  return image | im_floodfill_inv

def update(dummy=None):

  k10 = cv2.getTrackbarPos("K10", "edit")
  k11 = cv2.getTrackbarPos("K11", "edit")
  k20 = cv2.getTrackbarPos("K20", "edit")
  k21 = cv2.getTrackbarPos("K21", "edit")

  kernel1 = np.ones((1, 1), np.uint8)
  #kernel2 = np.ones((k20, k21), np.uint8)
  #kernel_smoth = np.ones((5, 5), np.uint8)

  #dil_iterations = cv2.getTrackbarPos("Dil_i", "edit")

  #dil_erosion = cv2.getTrackbarPos("Erode_i", "edit")

  dilation = cv2.dilate(img, kernel1, iterations = 2)
  im_floodfill = dilation.copy()
  h, w = im_floodfill.shape[:2]
  mask = np.zeros((h+2, w+2), np.uint8)
  cv2.floodFill(im_floodfill, mask, (1, 1), 255);
  im_floodfill_inv = cv2.bitwise_not(im_floodfill)
  #res = fill_holes(dilation)

  #smoth_image = cv2.filter2D(dilation, -1, kernel_smoth)
  #res = cv2.erode(dilation, kernel2, iterations = dil_erosion)
  cv2.imshow("edit", im_floodfill_inv)

cv2.namedWindow("edit")
cv2.createTrackbar("Dil_i", "edit", 1, 50, update)
#cv2.createTrackbar("Erode_i", "edit", 1, 50, update)
cv2.createTrackbar("K10", "edit", 1, 30, update)
cv2.createTrackbar("K11", "edit", 1, 30, update)
cv2.createTrackbar("K20", "edit", 1, 30, update)
cv2.createTrackbar("K21", "edit", 1, 30, update)
update()

while(1):

  k = cv2.waitKey(0) & 0xFF
  if k == 27: break
  update()

cv2.destroyAllWindows()