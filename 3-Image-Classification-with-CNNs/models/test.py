import os
import csv
import numpy as np

RESULTS = {
    'train': '../data/dermoscopyDBtrain.csv',
    'test': '../data/dermoscopyDBtest.csv',
    'val': '../data/dermoscopyDBval.csv'
}


def load_csvs(pred, test):
    output, real_output = [], []
    with open(pred) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for idx, img in enumerate(reader):
            output += [(idx + 2051, np.argmax(img))]

    with open(RESULTS[test]) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, img in enumerate(reader):
            if not i:
                continue
            real_output += [(int(img[0]), int(img[1]))]

    return output, real_output


def compute_acc(pred, test):
    output, real_output = load_csvs(pred, test)
    accuracy = 0
    for pred, real in zip(output, real_output):
        if pred[1] == real[1]:
            accuracy += 1
    accuracy /= 600
    return accuracy


def evaluate_outputs(test):
    accuracies = {}
    for model_res in os.listdir('.'):
        if model_res.endswith('csv'):
            accuracies[model_res] = compute_acc(model_res, test)
    print(sorted(accuracies.items(), key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    evaluate_outputs(test='test')
