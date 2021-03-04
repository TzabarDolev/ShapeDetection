import cv2
import numpy as np
from nms import non_max_suppression


def detect_circles(img):
    # load the image, clone it for output, and then convert it to grayscale
    print('detecting circles')
    height, width, channels = img.shape
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circle_fit = False
    accum = 1.2
    # detect circles in the image
    while not circle_fit:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, accum, 10)
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            circle_fit = True
        accum += 0.1
        if accum > 5:
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

def detect_lines(img):
    print('detecting triangles')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 10, 50)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
    lines_formatted = []
    try:
        lines = np.squeeze(lines, axis=1)
        lines_nms = non_max_suppression(lines, 0.3)
        for line in lines_nms:
            lines_formatted.append([[line[0], line[1]], [line[2], line[3]]])
    except:
        print('no lines detected')

    return lines_formatted

# separating between the circles and the triangles. circles are red and triangles are green (most of the times) so it
# will be easier to separate them and deal each one separately
def seperate_images(img):
    height, width, channels = img.shape

    # create 2 seperate images, one for circles and one for triangles
    imgr = np.zeros((height, width, 3), np.uint8)
    imgg = np.zeros((height, width, 3), np.uint8)

    # rgb_val will be the threshold for the seperation
    rgb_val = 120
    # seperating the images. making sure that the green is green and the red is red
    for i in range(height):
        for j in range(width):
            if img[i, j, 0] > rgb_val and img[i, j, 1] < rgb_val:
                imgr[i, j, :] = img[i, j]
                try:
                    # making the lines bolder. will make ROI a little worse but will make the detection better
                    imgr[i+1, j, :] = img[i, j]
                    imgr[i-1, j, :] = img[i, j]
                    imgr[i, j-1, :] = img[i, j]
                    imgr[i, j+1, :] = img[i, j]
                    imgr[i + 1, j+1, :] = img[i, j]
                    imgr[i - 1, j-1, :] = img[i, j]
                    imgr[i+1, j - 1, :] = img[i, j]
                    imgr[i-1, j + 1, :] = img[i, j]
                except:
                    print('one or more of the borders are at the boundaries of the image')

            if img[i, j, 1] > rgb_val and img[i, j, 0] < rgb_val:
                imgg[i, j, :] = img[i, j]
                try:
                    # making the lines bolder. will make ROI a little worse but will make the detection better
                    imgg[i + 1, j, :] = img[i, j]
                    imgg[i - 1, j, :] = img[i, j]
                    imgg[i, j - 1, :] = img[i, j]
                    imgg[i, j + 1, :] = img[i, j]
                    imgg[i + 1, j + 1, :] = img[i, j]
                    imgg[i - 1, j - 1, :] = img[i, j]
                    imgg[i + 1, j - 1, :] = img[i, j]
                    imgg[i - 1, j + 1, :] = img[i, j]
                except:
                    print('one or more of the borders are at the boundaries of the image')

    print('seperation complete')
    return imgr, imgg

