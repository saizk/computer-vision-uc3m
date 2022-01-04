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
import matplotlib.pyplot as plt

from skimage import io, filters, color, feature
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
from scipy import ndimage as nd

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


# -----------------------------------------------------------------------------
#
#     FUNCTIONS
#
# -----------------------------------------------------------------------------

# ----------------------------- Reading data function -------------------------
def read_data(data_dir=os.curdir, folder='train'):
    imgs_files = [os.path.join(data_dir, f'{folder}/images', f) for f in
                  sorted(os.listdir(os.path.join(data_dir, f'{folder}/images')))
                  if (os.path.isfile(os.path.join(data_dir, f'{folder}/images', f)) and f.endswith('.jpg'))]

    masks_files = [os.path.join(data_dir, f'{folder}/masks', f) for f in
                   sorted(os.listdir(os.path.join(data_dir, f'{folder}/masks')))
                   if (os.path.isfile(os.path.join(data_dir, f'{folder}/masks', f)) and f.endswith('.png'))]

    imgs_files = tuple(map(io.imread, imgs_files))
    masks_files = tuple(map(io.imread, masks_files))

    print(f"Number of {folder} images", len(imgs_files))
    print("Number of image masks", len(masks_files))

    return imgs_files, masks_files


def plot_img(img):
    plt.imshow(img, cmap='binary')
    plt.axis("off")


def plot_images(images, predicted_masks, gt_masks, scores):
    for idx, (img, pr_mask, gt_mask) in enumerate(zip(images, predicted_masks, gt_masks)):
        print(f'Image {idx + 1}')
        print(f'Jaccard Score {scores[idx]}')
        plot_imgs([img, pr_mask, gt_mask])


def plot_imgs(imgs):
    for idx, img in enumerate(imgs):
        plt.subplot(1, len(imgs), idx + 1)
        plot_img(img)
    plt.show()


# ----------------------------- Pre-processing function -------------------------


def hair_removal(image, params):
    kernel_size = cv2.getStructuringElement(1, (params.get('kernel'), params.get('kernel')))
    blackhat = cv2.morphologyEx(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
        cv2.MORPH_BLACKHAT,
        kernel=kernel_size
    )
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    filtered_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
    return filtered_image


def pre_processing(image, hr_params, filters_funcs):

    filtered_image = hair_removal(image, hr_params)
    filtered_image = color.rgb2hsv(filtered_image)[:, :, 1]

    for func, params in filters_funcs:
        filtered_image = func(filtered_image, **params)

    # p_low, p_high = np.percentile(filtered_image, (1, 95))
    # filtered_image = exp.rescale_intensity(filtered_image, in_range=(p_low, p_high))
    return filtered_image


# ----------------------------- Threshold functions -------------------------
def threshold_segmentation(image, params):

    threshold = thresholding(image, params.get('method'))

    if params.get('plot_thresh'):
        fig, ax = filters.try_all_threshold(image, figsize=(10, 8), verbose=False)
        plt.show()

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


# ----------------------------- Clustering function -------------------------
def cluster_segmentation(image, params):
    h, w = image.shape
    kmeans = KMeans(n_clusters=params.get('k'), random_state=0).fit(image.reshape(h * w, 1))
    cluster_centers = kmeans.cluster_centers_
    lower_level = min(np.unique(cluster_centers))
    predicted_mask_levels = cluster_centers[kmeans.labels_].reshape(h, w)
    predicted_mask = np.where(predicted_mask_levels == lower_level, 0, 1)
    return predicted_mask


# ----------------------------- Post-processing function -------------------------
def post_processing(image, filters_funcs):
    filtered_image = image
    for func, params in filters_funcs:
        filtered_image = func(filtered_image, **params)

    return filtered_image


# ----------------------------- Segmentation function -------------------------
def skin_lesion_segmentation(image, filters_funcs):
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
    filtered_image = pre_processing(image, filters_funcs['hair_removal'], filters_funcs['pre_processing'])

    if filters_funcs.get('thresh_segmentation'):
        predicted_mask = threshold_segmentation(filtered_image, filters_funcs['thresh_segmentation'])  # isodata

    elif filters_funcs.get('cluster_segmentation'):
        predicted_mask = cluster_segmentation(filtered_image, filters_funcs['cluster_segmentation'])

    else:
        raise Exception('Provide a segmentation method')

    predicted_mask = post_processing(predicted_mask, filters_funcs['post_processing'])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return predicted_mask


# ----------------------------- Jacc Score function ---------------------------
def compute_jacc_score(image, gt_mask, filters_funcs):
    predicted_mask = skin_lesion_segmentation(image, filters_funcs)
    gt_mask = gt_mask / 255
    jacc_score = jaccard_score(np.ndarray.flatten(gt_mask), np.ndarray.flatten(predicted_mask))
    return predicted_mask, gt_mask, jacc_score


# ----------------------------- Evaluation function ---------------------------
def evaluate_masks(images, gt_masks, filters_funcs):
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
    for i in np.arange(len(images)):
        print('I%d' % i)
        predicted_mask = skin_lesion_segmentation(images[i], filters_funcs)
        gt_mask = gt_masks[i] / 255
        score.append(jaccard_score(np.ndarray.flatten(gt_mask), np.ndarray.flatten(predicted_mask)))

    return np.mean(score)


def parallel_evaluation(images, gt_masks, filters_funcs,
                        plot_thresholds=False, plot_all=False,
                        plot_best_n=0, plot_worst_n=0):
    if plot_thresholds:
        filters_funcs['thresh_segmentation'].update({'plot_thresh': True})

    with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
        futures = [
            pool.submit(
                compute_jacc_score,
                images[i], gt_masks[i],
                filters_funcs
            )
            for i in np.arange(len(images))
        ]

        results = [f.result() for f in futures if f.result()]

    predicted_masks, gt_masks, scores = zip(*results)

    if plot_all:
        for idx, (image, pr_mask, gt_mask) in enumerate(zip(images, predicted_masks, gt_masks)):
            print(f'Image {idx + 1}')
            print(f'Jaccard Score {scores[idx]}')
            plot_imgs([image, pr_mask, gt_mask])

    if plot_best_n:
        first_n = sorted(list(enumerate(scores)), key=lambda x: x[1])[-plot_best_n:]
        for idx, score in first_n:
            print(f'Jacc Score: {score}')
            plot_imgs([images[idx], predicted_masks[idx], gt_masks[idx]])

    if plot_worst_n:
        last_n = sorted(list(enumerate(scores)), key=lambda x: x[1])[:plot_worst_n]
        for idx, score in last_n:
            print(f'Jacc Score: {score}')
            plot_imgs([images[idx], predicted_masks[idx], gt_masks[idx]])

    return np.mean(scores)


def main():
    st = time.time()
    print('Reading images...')
    train_imgs_files, train_masks_files = read_data(folder='test')

    filters_funcs = {
        'hair_removal': {'kernel': 3},
        'pre_processing': [
            (filters.median,              {'selem': morph.disk(30)}),  # (!)
            # (filters.rank.median,         {'selem': morph.disk(30)}),
            # (filters.unsharp_mask, {'radius': 0, 'amount': 2}),

            # (filters.rank.mean, {'selem': morph.disk(30)}),
            (filters.rank.geometric_mean, {'selem': morph.disk(30)}),  # (!)
            (filters.gaussian,            {'sigma': 11, 'multichannel': True}),  # (!)

            (exp.adjust_log,              {'gain': 1}),  # (!) 0.82449
            # (exp.equalize_adapthist,      {})  # bad
            # (filters.rank.mean_percentile, {'selem': morph.disk(20)})
        ],
        'thresh_segmentation': {
            'method': 'isodata'  # otsu
        },
        # 'cluster_segmentation': {
        #     'k': 2
        # },
        'post_processing': [
            (nd.binary_fill_holes, {}),
            (morph.erosion,           {'selem': morph.disk(5)}),
            (morph.dilation,          {'selem': morph.disk(30)}),
            # (morph.convex_hull_image, {'offset_coordinates': True})  # bad
        ]
    }

    # mean_score = evaluate_masks(train_imgs_files[:4], train_masks_files[:4], filters_funcs)
    mean_score = parallel_evaluation(
        train_imgs_files[:],
        train_masks_files[:],
        filters_funcs=filters_funcs,
        # plot_thresholds=True,
        # plot_all=True,
        # plot_best_n=1,
        # plot_worst_n=5
    )
    print(f'Mean Jacc score: {mean_score}')
    print(f'Elapsed time: {time.time() - st}')


if __name__ == '__main__':
    main()
