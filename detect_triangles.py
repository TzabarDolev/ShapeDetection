import cv2
import numpy as np

def detect_lines(img):
    print('detecting triangles')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
    lines_formatted = []
    try:
        lines = np.squeeze(lines, axis=1)
        for line in lines:
            lines_formatted.append([[line[0], line[1]], [line[2], line[3]]])
    except:
        print('no lines detected')

    return lines_formatted

