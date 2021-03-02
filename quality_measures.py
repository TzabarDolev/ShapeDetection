import json
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_json(json_file):
    # a function to read BB json files and return the BB dictionary
    file = open(json_file)
    file_BB = json.load(file)
    return file_BB


def extract_correctness_features(GT_grid, compare_grid):
    compare_dict = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for i in range(GT_grid.shape[0]):
        for j in range(GT_grid.shape[1]):
            if (GT_grid[i, j] == 1 and compare_grid[i, j] == 1) or (GT_grid[i, j] == 2 and compare_grid[i, j] == 2):
                compare_dict["TP"] += 1
            elif (GT_grid[i, j] == 1 or GT_grid[i, j] == 2) and compare_grid[i, j] == 0:
                compare_dict["FN"] += 1
            elif GT_grid[i, j] == 0 and (compare_grid[i, j] == 1 or compare_grid[i, j] == 2):
                compare_dict["FP"] += 1
            elif GT_grid[i, j] == 0 and compare_grid[i, j] == 0:
                compare_dict["TN"] += 1
    return compare_dict


def build_occupancy_grid(BB, shape):
    circles_bb = BB["circle"]
    triangle_bb = BB["triangle"]
    grid = np.zeros([shape[1], shape[0]])
    # filling BB occupancy grid
    for box in range(len(circles_bb)):
        for i in range(max(circles_bb[box][0][0], 0), min(circles_bb[box][1][0], shape[1])):
            for j in range(max(0, circles_bb[box][0][1]), min(circles_bb[box][1][1], shape[0])):
                grid[i, j] = 1
    for box in range(len(triangle_bb)):
        for i in range(max(0, triangle_bb[box][0][0]), min(triangle_bb[box][1][0], shape[1])):
            for j in range(max(0, triangle_bb[box][0][1]), min(triangle_bb[box][1][1], shape[0])):
                grid[i, j] = 2
    return grid


if __name__ == '__main__':
    files = os.listdir('img/')
    files.sort()
    pred_IOU, results_IOU = [], []
    pred_recall, results_recall = [], []
    pred_precision, results_precision = [], []
    pred_performance = {"image": [], "precision": [], "recall": [], "IOU": []}
    results_performance = {"image": [], "precision": [], "recall": [], "IOU": []}
    for img_path in tqdm(files):
        print('analyze image: ' + str(img_path[:-4]))
        img = cv2.imread('img/' + img_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # read BB json files
        GT_BB = read_json('ground_truth/' + img_path[:-4] + '.json')
        pred_BB = read_json('prediction/' + img_path[:-4] + '.json')
        results_BB = read_json('results/' + img_path[:-4] + '.json')

        # building occupancy grids for each category
        GT_grid = build_occupancy_grid(GT_BB, img.shape[:2])
        pred_grid = build_occupancy_grid(pred_BB, img.shape[:2])
        results_grid = build_occupancy_grid(results_BB, img.shape[:2])

        # extract correctness features: TP, FP, TN, FN
        GT_pred_compare = extract_correctness_features(GT_grid, pred_grid)
        GT_results_compare = extract_correctness_features(GT_grid, results_grid)

        # measuring IOU, precision and recall
        # sometimes it's zero so i added this to the safe side
        results_precision.append(GT_results_compare["TP"] / (GT_results_compare["TP"] + GT_results_compare["FP"] + 0.001))
        pred_precision.append(GT_pred_compare["TP"] / (GT_pred_compare["TP"] + GT_pred_compare["FP"]))
        pred_recall.append(GT_pred_compare["TP"] / (GT_pred_compare["TP"] + GT_pred_compare["FN"]))
        results_recall.append(GT_results_compare["TP"] / (GT_results_compare["TP"] + GT_results_compare["FN"]))
        pred_IOU.append(GT_pred_compare["TP"] / (GT_pred_compare["TP"] + GT_pred_compare["FN"] + GT_pred_compare["FP"]))
        results_IOU.append(GT_results_compare["TP"] / (GT_results_compare["TP"] + GT_results_compare["FN"] + GT_results_compare["FP"]))

        # save information for topk, worstk
        # [image, precision, recall, IOU]
        pred_performance["image"].append(str(img_path[:-4]))
        pred_performance["precision"].append(GT_pred_compare["TP"] / (GT_pred_compare["TP"] + GT_pred_compare["FP"]))
        pred_performance["recall"].append(GT_pred_compare["TP"] / (GT_pred_compare["TP"] + GT_pred_compare["FN"]))
        pred_performance["IOU"].append(GT_pred_compare["TP"] / (GT_pred_compare["TP"] + GT_pred_compare["FN"] + GT_pred_compare["FP"]))
        results_performance["image"].append(str(img_path[:-4]))
        results_performance["precision"].append(GT_results_compare["TP"] / (GT_results_compare["TP"] + GT_results_compare["FP"] + 0.0001))
        results_performance["recall"].append(GT_results_compare["TP"] / (GT_results_compare["TP"] + GT_results_compare["FN"]))
        results_performance["IOU"].append(GT_results_compare["TP"] / (GT_results_compare["TP"] + GT_results_compare["FN"] + GT_results_compare["FP"]))

    output_score = {"prediction IOU": np.mean(pred_IOU), "results IOU": np.mean(results_IOU),
                    "prediction recall": np.mean(pred_recall), "results recall": np.mean(results_recall),
                    "prediction precision": np.mean(pred_precision), "results precision": np.mean(results_precision)}
    with open('assets/output_score.txt', 'w') as file:
        file.write(json.dumps(output_score))

    pred_performance = pd.DataFrame(pred_performance)
    pred_performance.to_csv('assets/pred_performance.txt')
    results_performance = pd.DataFrame(results_performance)
    results_performance.to_csv('assets/results_performance.txt')

