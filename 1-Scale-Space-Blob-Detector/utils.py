# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def nms(filter_response, min_distance, threshold):
    """
   from Svetlana Lazebnik, University of Illinois at Urbana-Champaign (Computer Vision) 
   """

    # find top candidates above a threshold
    filter_response_t = (filter_response > threshold) * 1

    # get coordinates of candidates
    candidates = filter_response_t.nonzero()
    coords = [(candidates[0][c], candidates[1][c]) for c in range(len(candidates[0]))]
    # ...and their values
    candidate_values = [filter_response[c[0]][c[1]] for c in coords]
    # sort candidates
    index = np.argsort(candidate_values)
    index = np.flip(index)
    # print('index: ', index)
    # store allowed point locations in array
    allowed_locations = np.zeros(filter_response.shape)
    allowed_locations[min_distance:-min_distance, min_distance:-min_distance] = 1
    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i][0] - min_distance):(coords[i][0] + min_distance),
            (coords[i][1] - min_distance):(coords[i][1] + min_distance)] = 0

    return filtered_coords


def plot_circles(image, cx, cy, rad):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    from Svetlana Lazebnik, University of Illinois at Urbana-Champaign (Computer Vision)
    """

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy, rad):
        circ = Circle((y, x), r, color="r", linewidth=1.5, fill=False)
        ax.add_patch(circ)

    plt.title(f'{len(cx)} circles')
    plt.show()
