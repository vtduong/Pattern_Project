import math
import numpy as np
import cv2
from copy import deepcopy
from timer import timing
from skimage.measure import label
import sys

np.set_printoptions(threshold=sys.maxsize)

EPSILON = 1  # variable for smooth_cost


def get_neighbour_indices(x, y, x_max, y_max):
    n = []
    if x > 0:
        n.append([x - 1, y])
    if x < x_max:
        n.append([x + 1, y])
    if y > 0:
        n.append([x, y - 1])
    if y < y_max:
        n.append([x, y + 1])
    return n


def smooth_cost_set(indices, img):
    cost = 0
    for x, y in zip(indices[0], indices[1]):
        n_indices = get_neighbour_indices(x, y, img.shape[0] - 1, img.shape[1] - 1)
        neighbours = [img[n[0], n[1]] for n in n_indices]
        cost += sum([smooth_cost(img[x, y], n) for n in neighbours])
    return cost
    # get_neighbours(i)


def smooth_cost(p1, p2):
    if p1 != p2:
        return EPSILON
    else:
        return 0


def data_cost(hashed_img):
    count = np.array(
        [len(hashed_img[np.where(hashed_img == i)]) for i in np.unique(hashed_img)]
    )
    freq = count / sum(count)
    return -np.log(freq) / np.log(2), np.unique(hashed_img)  # log base 2


def find_colour_sets(hashed_img):  # equivalent of smooth cost
    return label(hashed_img, connectivity=1)


@timing
def bad_alpha_expansion(hashed_img):
    # for each main colour, get data_cost
    dcost, dcost_indices = data_cost(hashed_img)
    # find the contiguous colour sets
    colour_sets = find_colour_sets(hashed_img)
    # get their indices
    cs_indices = [np.where(colour_sets == cs) for cs in np.unique(colour_sets)]
    # fast alpha expansion
    print(len(cs_indices))
    for cs_set in cs_indices:
        cs_set_col = hashed_img[cs_set][0]

        # current data_cost + smooth_cost of first region
        r1_cost = dcost[np.where(dcost_indices == cs_set_col)[0]] + smooth_cost_set(
            cs_set, hashed_img
        )

        # for each point set, calculate their own cost.
        for scd_cs_set in cs_indices:
            scd_cs_set_col = hashed_img[scd_cs_set][0]
            if cs_set is scd_cs_set:
                continue  # skip
            else:
                # current cost of first region + second region
                r2_cost = dcost[
                    np.where(dcost_indices == scd_cs_set_col)[0]
                ] + smooth_cost_set(scd_cs_set, hashed_img)
                current_cost = r1_cost + r2_cost

                # what if first region was set to the colour of the second ? calculate cost
                hashed_copy = deepcopy(hashed_img)
                hashed_copy[cs_set] = scd_cs_set_col
                r1_changed_cost = dcost[
                    np.where(dcost_indices == scd_cs_set_col)[0]
                ] + smooth_cost_set(cs_set, hashed_copy)
                r2_changed_cost = dcost[
                    np.where(dcost_indices == scd_cs_set_col)[0]
                ] + smooth_cost_set(scd_cs_set, hashed_copy)

                changed_cost = r1_changed_cost + r2_changed_cost
                # if cost is less, change colour of region
                if changed_cost < current_cost:
                    hashed_img = hashed_copy
                    # recalculate dcost
                    dcost, dcost_indices = data_cost(hashed_img)
                    # find the contiguous colour sets
                    colour_sets = find_colour_sets(hashed_img)

    print(np.unique(hashed_img))
    print(hashed_img)
    return hashed_img
