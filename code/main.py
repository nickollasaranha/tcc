import cv2
import numpy as np
import math

# this returns in degrees
def horizontal_angle(line):
    return 0 if line[2]-line[0] == 0 else math.degrees(math.atan(((line[3]-line[1])/(line[2]-line[0]))))

# pixels
def euclid_dist(a, b):
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

def line_width(line):
    return math.sqrt((line[2]-line[0])**2 + (line[3]-line[1])**2)

def filter_plate_lines(lines, angle_interval):

    new_lines = []
    vertical_width_range = [15, 50]
    horizontal_width_range = [15, 100]

    for line in lines:

        # Vertical stuff
        if (angle_interval[1][0] <= abs(horizontal_angle(line[0])) <= angle_interval[1][1]) and ((vertical_width_range[0] <= line_width(line[0]) <= vertical_width_range[1])):
            new_lines.append(line)
        # Horizontal stuff
        elif (angle_interval[0][0] <= abs(horizontal_angle(line[0])) <= angle_interval[0][1]) and ((horizontal_width_range[0] <= line_width(line[0]) <= horizontal_width_range[1])):
            new_lines.append(line)

    return np.array(new_lines) 

def draw(lines):
    img = np.zeros((1080, 1920, 3), np.uint8)
    drawn_img = lsd.drawSegments(img, lines) 
    #resized = cv2.resize(drawn_img, (0,0), fx=10, fy=10)
    cv2.imshow('image', drawn_img)
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

img = cv2.imread('track2.png', 0)

canny = auto_canny(img)

# Detect lines in the image
# lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE, 1)
# lines = lsd.detect(img)
# lines = filter_plate_lines(lines[0], [[0, 10], [85, 95]])
# img = np.zeros((1080, 1920, 3), np.uint8)
# drawn_img = lsd.drawSegments(img, lines)

dilated_image = cv2.dilate(canny, np.ones((1, 3), np.uint8), iterations = 4)

# mask = np.array([
#         [0, 0, 0],
#         [1, 1, 1],
#         [0, 0, 0]
#     ])

# filtered_image = cv2.filter2D(dilated_image, -1, mask)

#dilated_image = cv2.dilate(gray_image, mask, iterations = 1)
#

#im2, contours, hierarchy = cv2.findContours(dilated_image.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#for cnt in contours: cv2.drawContours(im2, [cnt], 0, 255, -1)

cv2.imshow('image', dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# img = np.zeros((1080, 1920, 3), np.uint8)
# drawn_img = lsd.drawSegments(img, lines)

# gray_image = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2GRAY)
# dilated_image = cv2.dilate(gray_image, np.ones((1, 5), np.uint8), iterations = 2)

# ret, thresh = cv2.threshold(dilated_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# for cnt in contours: cv2.drawContours(im2, [cnt], 0, 255, -1)

# cv2.destroyAllWindows()

# kernel = np.ones((3, 7), np.uint8)
# eroded = cv2.erode(im2, kernel, iterations = 3)
# #moth_image = cv2.filter2D(im2, -1, kernel_smoth)

# cv2.imshow('image', drawn_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#draw(lines)
# for line in horizontal_lines:

#    print (line)
#    #drawn_img = lsd.drawSegments(roi, np.array(line)) 
#    draw(np.array(line))

#TODO:
# Try to remove horizontal segments between bla bla
# Try to remove vertical segments higher than bla bla
# Create a func to check for vertical distances between horizontal lines.
# Create a func to check for horizontal distances between vertical lines.
# Wrap it all and done.