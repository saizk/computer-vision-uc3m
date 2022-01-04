#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %      
%   Audio Processing, Video Processing and Computer Vision              %
%                                                                       %
%   LS3: SEGMENTATION OF PIGMENTED SKIN LESIONS                         %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import os
import cv2
import time
import numpy as np
import skimage.exposure as exp
import skimage.morphology as morph

from skimage import io, filters, color, feature
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
from scipy import ndimage as nd

from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor


# -----------------------------------------------------------------------------
#
#     FUNCTIONS
#
# -----------------------------------------------------------------------------

# ----------------------------- Hair removal function -------------------------
def hair_removal(image):
    blackhat = cv2.morphologyEx(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
        cv2.MORPH_BLACKHAT,
        kernel=cv2.getStructuringElement(1, (3, 3))
    )
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    filtered_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
    return filtered_image


# ----------------------------- Pre-processing function -------------------------
def pre_processing(image):
    filtered_image = hair_removal(image)

    filtered_image = color.rgb2hsv(filtered_image)[:, :, 1]  # saturation channel

    filtered_image = filters.median(filtered_image, selem=morph.disk(20))
    filtered_image = filters.rank.geometric_mean(filtered_image, selem=morph.disk(20))
    filtered_image = filters.gaussian(filtered_image, sigma=11, multichannel=True)

    filtered_image = exp.adjust_log(filtered_image, 1)
    return filtered_image


# ----------------------------- Thresholding functions -------------------------
def threshold_segmentation(image, method='isodata'):
    threshold = thresholding(image, method)
    predicted_mask = (image > threshold).astype('int')
    return predicted_mask


def thresholding(image, method):
    if method == 'otsu':
        threshold = filters.threshold_otsu(image)
    elif method == 'isodata':
        threshold = filters.threshold_isodata(image)
    elif method == 'yen':
        threshold = filters.threshold_yen(image)
    elif method == 'mean':
        threshold = filters.threshold_mean(image)
    elif method == 'triangle':
        threshold = filters.threshold_triangle(image)
    elif method == 'li':
        threshold = filters.threshold_li(image)
    else:
        raise Exception('Unknown method')

    return threshold


# ----------------------------- Clustering functions ------------------------

def cluster_segmentation(image, k=2):
    h, w = image.shape
    kmeans = KMeans(n_clusters=k, random_state=0).fit(image.reshape(h * w, 1))

    cluster_centers = kmeans.cluster_centers_
    lower_level = min(np.unique(cluster_centers))

    predicted_mask_levels = cluster_centers[kmeans.labels_].reshape(h, w)
    predicted_mask = np.where(predicted_mask_levels == lower_level, 0, 1)

    return predicted_mask


# ----------------------------- Post-processing function -------------------------
def post_processing(image):
    filtered_image = nd.binary_fill_holes(image)
    filtered_image = morph.erosion(filtered_image, selem=morph.disk(5))
    filtered_image = morph.dilation(filtered_image, selem=morph.disk(30))

    return filtered_image


# ----------------------------- Segmentation function -------------------------
def skin_lesion_segmentation(img_root):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                       %
    %  Template for implementing the main function of the segmentation      %
    % system: 'skin_lesion_segmentation'                                    %
    % - Input: file name of the image showing the skin lesion               %
    % - Output: predicted segmentation mask                                 %
    %                                                                       %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    SKIN_LESION_SEGMENTATION:
    - - -  COMPLETE - - -
    """
    # Code for the BASELINE system
    image = io.imread(img_root)
    filtered_image = pre_processing(image)

    predicted_mask = threshold_segmentation(filtered_image, method='isodata')  # isodata
    # predicted_mask = cluster_segmentation(filtered_image, k=2)

    predicted_mask = post_processing(predicted_mask)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return predicted_mask


# ----------------------------- Evaluation function ---------------------------
def evaluate_masks(img_roots, gt_masks_roots):
    """ EVALUATE_MASKS: 
        It receives two lists:
        1) A list of the file names of the images to be analysed
        2) A list of the file names of the corresponding Ground-Truth (GT) 
            segmentations
        For each image on the list:
            It performs the segmentation
            It determines Jaccard Index
        Finally, it computes an average Jaccard Index for all the images in 
        the list
    """
    score = []
    for i in np.arange(np.size(img_roots)):
        print('I%d' % i)
        predicted_mask = skin_lesion_segmentation(img_roots[i])
        gt_mask = io.imread(gt_masks_roots[i]) / 255
        score.append(jaccard_score(np.ndarray.flatten(gt_mask), np.ndarray.flatten(predicted_mask)))
    mean_score = np.mean(score)
    print('Average Jaccard Index: ' + str(mean_score))
    return mean_score


# -------------------------- Parallel evaluation function ---------------------------
def compute_jacc_score(img, gt_mask):
    predicted_mask = skin_lesion_segmentation(img)
    gt_mask = io.imread(gt_mask) / 255
    jacc_score = jaccard_score(np.ndarray.flatten(gt_mask), np.ndarray.flatten(predicted_mask))
    return jacc_score


def parallel_evaluation(img_roots, gt_masks_roots, plot=False):
    with ThreadPoolExecutor(max_workers=cpu_count() - 1) as pool:
        futures = [
            pool.submit(
                compute_jacc_score,
                img_roots[i], gt_masks_roots[i]
            )
            for i in np.arange(np.size(img_roots))
        ]

        scores = [f.result() for f in futures if f.result()]

    mean_score = np.mean(scores)
    return mean_score


# -----------------------------------------------------------------------------
#
#     READING IMAGES
#
# -----------------------------------------------------------------------------
def read_data(data_dir=os.curdir, folder='train'):
    train_imgs_files = [os.path.join(data_dir, f'{folder}/images', f) for f in
                        sorted(os.listdir(os.path.join(data_dir, f'{folder}/images')))
                        if (os.path.isfile(os.path.join(data_dir, f'{folder}/images', f)) and f.endswith('.jpg'))]

    train_masks_files = [os.path.join(data_dir, f'{folder}/masks', f) for f in
                         sorted(os.listdir(os.path.join(data_dir, f'{folder}/masks')))
                         if (os.path.isfile(os.path.join(data_dir, f'{folder}/masks', f)) and f.endswith('.png'))]

    # train_imgs_files.sort()
    # train_masks_files.sort()
    print("Number of train images", len(train_imgs_files))
    print("Number of image masks", len(train_masks_files))

    return train_imgs_files, train_masks_files


# -----------------------------------------------------------------------------
#
#     Segmentation and evaluation
#
# -----------------------------------------------------------------------------

def main():
    st = time.time()
    train_imgs_files, train_masks_files = read_data(folder='train')

    # mean_score = evaluate_masks(train_imgs_files, train_masks_files)
    mean_score = parallel_evaluation(
        tuple(train_imgs_files), tuple(train_masks_files),
        plot=False
    )
    print(f'Mean Jacc score: {mean_score}')
    print(f'Elapsed time: {time.time() - st}')


if __name__ == '__main__':
    main()
