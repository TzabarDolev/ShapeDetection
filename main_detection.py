import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from shape_detection import detect_lines, detect_circles, seperate_images


# a json class that let's me write json files
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def locate_boundaries(img, img_path):
    print('seperate circles and triangles')
    circles, triangles = seperate_images(img)
    circle_location_list = detect_circles(img)
    print('circles locations:' + str(circle_location_list))
    triangle_location_list = detect_lines(img)
    print('triangles locations:' + str(triangle_location_list))

    results = {}
    results["circle"] = []
    for circle in range(len(circle_location_list)):
        results["circle"].append(circle_location_list[circle])
    results["triangle"] = []
    for triangle in range(len(triangle_location_list)):
        results["triangle"].append(triangle_location_list[triangle])

    filename = 'results/' + img_path[:-4] + '.json'
    with open(filename, 'w') as outfile:
        json.dump(results, outfile, cls=NpEncoder)

    print('results extracted successfully: ' + str(img_path[:-4]))

if __name__ == '__main__':
    files = os.listdir('img/')
    for img_path in tqdm(files):
        print('analyze image: ' + str(img_path[:-4]))
        img = cv2.imread('img/' + img_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locate_boundaries(img, img_path)

