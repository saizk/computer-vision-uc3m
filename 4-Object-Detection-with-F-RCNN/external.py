#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:22:39 2020

@author: mmolina
"""
import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

min_size = 800
max_size = 1333
result_dir = 'baseline_results'


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_gt_anchors(data_loader_0):
    # Anchor statistics
    ar = []
    labels = []
    size_anchors = []
    for data, target, img_path in data_loader_0:
        size_img = data[0].size()[1:]
        scale_factor = min_size / min(size_img)
        if max(size_img) * scale_factor > max_size:
            scale_factor = max_size / max(size_img)
        boxes = target[0]['boxes'].cpu().numpy() * scale_factor
        l_t = target[0]['labels'].cpu().numpy()
        for i in range(boxes.shape[0]):
            labels.append(l_t[i])
            size_anchors.append(np.sqrt((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])))
            ar.append((boxes[i, 2] - boxes[i, 0]) / (boxes[i, 3] - boxes[i, 1]))
    ar = np.array(ar)
    labels = np.array(labels)
    size_anchors = np.array(size_anchors)
    return ar, labels, size_anchors


def get_features_hook(self, input, output):
    if not os.path.exists(os.path.join(result_dir, 'aux')):
        os.mkdir(os.path.join(result_dir, 'aux'))
    features = output.data.cpu().numpy()
    np.save(os.path.join(result_dir, 'aux', 'features.npy'), features)


def visualize_roipooling(model, dataloaders, device, class_names, th_score, th_iou, result_dir, SAVE_OPT, batch_size=1):
    model.eval()  # Set model to evaluate mode
    model.roi_heads.box_roi_pool.register_forward_hook(get_features_hook)

    print('Evaluating...')
    with torch.no_grad():
        # Iterate over data.
        for inputs, targets, paths in dataloaders:
            inputs = list(image.to(device) for image in inputs)
            labels = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if np.array(labels[0]['boxes'].cpu()).shape[0] == 0:
                gt_boxes = np.zeros((0, 4), dtype='float32')
            else:
                gt_boxes = np.array(labels[0]['boxes'].detach().cpu())
                gt_labels = np.array(labels[0]['labels'].detach().cpu())

            pred = model(inputs)
            pred[0]['scores'] = pred[0]['scores'].detach().cpu()
            pred[0]['boxes'] = pred[0]['boxes'].detach().cpu()
            pred[0]['labels'] = pred[0]['labels'].detach().cpu()
            if len(pred[0]['scores'].numpy()) == 0:
                pred_boxes = np.zeros((0, 4), dtype='float32')
                pred_labels = np.zeros((0,), dtype=int)
            else:
                pred_score = list(pred[0]['scores'].numpy())
                if pred_score[0] > th_score:
                    pred_t = [pred_score.index(x) for x in pred_score if x > th_score][-1]
                    pred_class = [class_names[i] for i in list(pred[0]['labels'].numpy())]
                    pred_labels = pred[0]['labels'].numpy()
                    pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].numpy())]
                    pred_boxes = np.array(pred_boxes[:pred_t + 1])
                    pred_class = pred_class[:pred_t + 1]
                    pred_labels = pred_labels[:pred_t + 1]
                else:
                    pred_boxes = np.zeros((0, 4), dtype='float32')
                    pred_labels = np.zeros((0,), dtype=int)

            # Load ROI features
            features = np.load(os.path.join(result_dir, 'aux', 'features.npy'))
            features = np.max(features, axis=1)
            features = features[pred[0]['ids'].cpu(), :, :]
            if len(features.shape) < 3:
                features = features[None, :, :]
            aux = paths[0].split('/')
            folder_path = os.path.join(result_dir, 'roipool')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            img = np.array(F.to_pil_image(inputs[0].cpu()))
            for j in range(len(pred_labels)):
                ious = np.zeros((gt_boxes.shape[0],), dtype='float')
                for k in range(gt_boxes.shape[0]):
                    ious[k] = bb_intersection_over_union(gt_boxes[k, :], pred_boxes[j, :])
                if len(ious) > 0:
                    iou_max = np.max(ious)
                    pos = np.argmax(ious)
                    if (iou_max > th_iou) and (gt_labels[pos] == pred_labels[j]):
                        pred_path = folder_path + '/' + aux[-1][:-4] + '_' + str(j) + '_' + str(pred_labels[j]) + '.png'
                        roi = img[int(pred_boxes[j][1]):int(pred_boxes[j][3]),
                                  int(pred_boxes[j][0]):int(pred_boxes[j][2]), :]
                        cv2.imwrite(pred_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
                        roi = cv2.resize((255.0 * ((features[j, :, :] - np.min(features[j, :, :])) / np.max(
                            features[j, :, :]))).astype(np.uint8), (roi.shape[1], roi.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                        pred_path = folder_path + '/' + aux[-1][:-4] + '_' + str(j) + '_' + str(
                            pred_labels[j]) + '_roi.png'
                        cv2.imwrite(pred_path, roi)
