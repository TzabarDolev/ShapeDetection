import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display_BB(img, json, color, thickness):
    circles_bb = json["circle"]
    triangle_bb = json["triangle"]
    for bb in circles_bb:
        img = cv2.rectangle(img, tuple(bb[0]), tuple(bb[1]), color, thickness)
        cv2.putText(img, 'circle', tuple(bb[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

    for bb in triangle_bb:
        img = cv2.rectangle(img, tuple(bb[0]), tuple(bb[1]), color, thickness)
        cv2.putText(img, 'triangle', tuple(bb[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    return img


def read_json(json_file):
    # a function to read BB json files and return the BB dictionary
    file = open(json_file)
    file_BB = json.load(file)
    return file_BB


def show_topk(df, k):
    # getting k top and worst predictions
    IOU_worst = df.nsmallest(k, 'IOU')
    precision_worst = df.nsmallest(k, 'precision')
    recall_worst = df.nsmallest(k, 'recall')
    IOU_top = df.nlargest(k, 'IOU')
    precision_top = df.nlargest(k, 'precision')
    recall_top = df.nlargest(k, 'recall')

    series = [IOU_worst, IOU_top, precision_worst, precision_top, recall_worst, recall_top]
    titles = ['IOU_worst_k', 'IOU_top_k', 'precision_worst_k', 'precision_top_k', 'recall_worst_k', 'recall_top_k']

    img_path = 'img/'
    for data in range(len(series)):
        plt.figure(figsize=[20, 5])
        plt.title(str(titles[data]))
        for plot in range(k):
            # read bounding boxes
            results_BB = read_json('results/' + str(series[data]["image"].values[plot]) + '.json')
            plt.subplot(1, k, plot+1)
            img = cv2.imread(img_path + str(series[data]["image"].values[plot]) + str('.jpg'))
            # adding boung box
            img = display_BB(img, results_BB, (0, 255, 0), 2)
            plt.axis('off')
            plt.imshow(img)
            if data == 0 or data == 1:
                type = 'IOU'
            elif data == 2 or data == 3:
                type = 'precision'
            else:
                type = 'recall'
            plt.title(str(series[data][type].values[plot]))
        # plt.show()
        plt.savefig('assets/' + str(titles[data]) + '.jpg')


if __name__ == '__main__':
    file_results = 'assets/results_performance.txt'
    file_pred = 'assets/pred_performance.txt'
    df_results = pd.read_csv(file_results)
    df_pred = pd.read_csv(file_pred)
    k_picks = 5
    show_topk(df_results, k_picks)

    # plotting graphs
    x = np.linspace(1, 500, 500)
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle('IOU comparison')
    axs[0].plot(x, df_results["IOU"].values)
    axs[0].set_title('my results IOU')
    axs[1].plot(x, df_pred["IOU"].values)
    axs[1].set_title('prediction IOU')
    plt.savefig('assets/' + 'IOU_comparison.jpg')

    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle('precision comparison')
    axs[0].plot(x, df_results["precision"].values)
    axs[0].set_title('my results precision')
    axs[1].plot(x, df_pred["precision"].values)
    axs[1].set_title('prediction precision')
    plt.savefig('assets/' + 'precision_comparison.jpg')

    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle('recall comparison')
    axs[0].plot(x, df_results["recall"].values)
    axs[0].set_title('my results recall')
    axs[1].plot(x, df_pred["recall"].values)
    axs[1].set_title('prediction recall')
    plt.savefig('assets/' + 'recall_comparison.jpg')
    plt.show()
