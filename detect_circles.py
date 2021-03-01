import cv2
import numpy as np


def detect_circles(img):
    # load the image, clone it for output, and then convert it to grayscale
    print('detecting circles')
    height, width, channels = img.shape
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circle_fit = False
    accum = 1
    # detect circles in the image
    while not circle_fit:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, accum, 10)
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            circle_fit = True
        accum += 0.05
        if accum > 25:
            circle_fit = True
    circle_location_list = find_BoundingBox(circles, img.shape[:2])
    return circle_location_list


def find_BoundingBox(circles, shape):
    circle_location_list = []
    try:
        for (x, y, r) in circles:
            # making sure that bounds do not exceed image properties
            if x-r < 0:
                upper_left_x = 0
            else:
                upper_left_x = x - r
            if y-r < 0:
                upper_left_y = 0
            else:
                upper_left_y = y - r
            if x+r > shape[0]:
                lower_right_x = shape[0]
            else:
                lower_right_x = x + r
            if y+r > shape[1]:
                lower_right_y = shape[1]
            else:
                lower_right_y = y + r
            circle_location_list.append([[upper_left_x, upper_left_y], [lower_right_x, lower_right_y]])
    except:
        print('no circles detected')

    return circle_location_list

