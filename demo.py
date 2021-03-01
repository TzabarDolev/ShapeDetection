import json
import os
import random

import cv2
import matplotlib.pyplot as plt


def read_json(json_file):
    # a function to read BB json files and return the BB dictionary
    file = open(json_file)
    file_BB = json.load(file)
    return file_BB

def display_BB(img, json, color, thickness):
    circles_bb = json["circle"]
    triangle_bb = json["triangle"]
    for bb in circles_bb:
        img = cv2.rectangle(img, tuple(bb[0]), tuple(bb[1]), color, thickness)
        cv2.putText(img, 'circle', tuple(bb[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        plt.imshow(img)

    for bb in triangle_bb:
        img = cv2.rectangle(img, tuple(bb[0]), tuple(bb[1]), color, thickness)
        cv2.putText(img, 'triangle', tuple(bb[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        plt.imshow(img)
    plt.axis('off')
    plt.show()
    return img

# reading all images and choosing a random image to show
files = os.listdir('img/')
sample = random.randint(0, len(files))
print('analyze image: ' + str(files[sample]))

# reading the image
img_filename = 'img/' + files[sample]
img = cv2.imread(img_filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# reading the ground truth and prediction
GT_BB = read_json('ground_truth/' + files[sample][:-4] + '.json')
pred_BB = read_json('prediction/' + files[sample][:-4] + '.json')

# will work only after running the main_detection.py file
# results_BB = read_json('results/' + files[sample][:-4] + '.json')


# showing the image
fig = plt.figure()
plt.imshow(img)
plt.axis('off')
plt.show()

# showing the image with GT
img = display_BB(img, GT_BB, (255, 0, 0), 2)

# showing the image with GT and prediction
img2 = display_BB(img, pred_BB, (0, 255, 0), 1)

# showing the image with GT and my results
# img3 = img2 = display_BB(img, results_BB, (0, 0, 255), 1)

