import numpy as np
import itertools
from timer import timing
from copy import deepcopy
import graph_cuts as gc

np.set_printoptions(suppress=True)


def get_pixel_list(img):
    """Given an x by y by c img, returns a x*y by c array, and the original shape.
    Essentially, gets the img as a list of pixels, each pixel being a vector of
    length c. """
    orig_shape = img.shape  # Remember the original shape of the img.
    # Store the img as a x by z array (z being the length of the colour space)
    # Essentially just a list of pixels.

    if len(img.shape) == 3:
        img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    elif len(img.shape) == 2:
        img = img.reshape(img.shape[0] * img.shape[1],)
    return orig_shape, img


def sort_by_rows(arr):
    """Sort an array by its rows."""
    return arr[np.lexsort(arr.T[::-1])]


def get_bin_lims(n, max_value):
    """Given a n and a max_value, returns bin separation values"""
    return np.linspace(max_value // n, max_value, n, dtype=int)


def hash_pixel(p, n, max_value):
    """ Given a vector p of length c and a list of bin limits, hashes the pixel
    in a particular bin"""
    multiplier = np.flip(np.array([2] * len(p)) ** range(0, len(p)))
    return sum(p // ((max_value // n) + 1) * multiplier)


# for debugging
def get_bins(size, n, max_value):
    """Given a number of channels (size), a number of buckets per channel (n),
    and a max value, returns all the possible bins.
    The bins are stored in the following manner:
        Supposing we are working on the RGB colour space (max_value = 255),
        with 3 channels (size = 3), with 2 buckets per channel (n = 2)
        Then I have the following bins:
        [[0-127], [0-127], [0-127],  (0th bin)
        [0-127], [0-127], [128-255], (1st bin)
        [0-127], [127-255], [0-127], (2nd bin)
        ...
        They will be returned to you in the following format:
        [[127, 127, 127],
         [127, 127, 255],
         [127, 255, 127],
         ...]
        """
    bin_lims = get_bin_lims(n, max_value)
    return sort_by_rows(np.array(list(itertools.product(bin_lims, repeat=size))))


def get_bin(index, size, n, max_value):
    lims = get_bin_lims(n, max_value)
    num_of_bins = n ** size
    splits = num_of_bins // len(lims)
    bin = []
    for i in range(size):
        bin.append(lims[index // splits])
        index -= (index // splits) * splits
        splits = splits // len(lims)
    return bin


# given a bin, get the neighbouring bins, with respect to colours
def get_nearby_bins(bin, lims):
    neighbours = []
    for i in range(len(bin)):
        l_i = np.where(lims == bin[i])[0][0]
        if not l_i == 0:
            b = deepcopy(bin)
            b[i] = lims[l_i - 1]
            neighbours.append(b)
        if not l_i == len(lims) - 1:
            b = deepcopy(bin)
            b[i] = lims[l_i + 1]
            neighbours.append(b)
    return neighbours


@timing
def hash_img(img, n=3, max_value=255):
    """Given an x by y by c img, a number (n) of buckets per channel (c),
    and a max value (in the case of RGB, 255), returns an x by y
    array denoting the bin indices for each pixel. """
    orig_shape, img = get_pixel_list(img)

    # Note: apply along axis is slower, but this is still slow.
    # There probably exists a way of making this faster with numpy.

    # return np.apply_along_axis(hash_pixel, 1, img, n, max_value).reshape(orig_shape[:2])
    return np.array([hash_pixel(p, n, max_value) for p in img]).reshape(orig_shape[:2])


def get_pixel_in_this_bin(hashed_img, i):
    return np.where(hashed_img == i)


def get_mean_colour_per_bin(img, hashed_img):
    """Given an img and its hashed equivalent, returns, for each bin, the average colour."""
    # this function is very fast, no need to optimise it
    _, img = get_pixel_list(img)
    hashed_img = hashed_img.reshape(img.shape[:1])
    # for each bin i, get the indices of all pixels hashed in that bin, and get the mean of these
    return np.array(
        [np.mean(img[np.where(hashed_img == i)], axis=0) for i in np.unique(hashed_img)]
    )


def make_img_with_assigned_colours(colours, hashed_img):
    """Given a list of colours and a list of pixels assigned to each colour,
    returns list of pixels with correctly assigned colours."""
    orig_shape = hashed_img.shape
    hashed_img = hashed_img.flatten()
    img = np.ones((*hashed_img.shape, colours.shape[1],), dtype="uint8")
    for i in range(len(colours)):
        img[np.where(hashed_img == i)] = colours[i]
    return img.reshape((*orig_shape, len(colours[0])))


@timing
def filter_low_frequency_colours(hashed_img, threshhold=0.075):
    for i in range(100):
        original = deepcopy(hashed_img)
        count = np.array(
            [len(hashed_img[np.where(hashed_img == i)]) for i in np.unique(hashed_img)]
        )
        freq = count / sum(count)
        for i, f in enumerate(freq):
            if f < threshhold:
                concerned_pixels = np.where(hashed_img == i)
                for x, y in zip(concerned_pixels[0], concerned_pixels[1]):
                    n_indices = gc.get_neighbour_indices(
                        x, y, hashed_img.shape[0] - 1, hashed_img.shape[1] - 1
                    )
                    neighbours = [hashed_img[n[0], n[1]] for n in n_indices]
                    fi = np.where(freq == max([freq[n] for n in neighbours]))[0]
                    print(fi)
                    hashed_img[x, y] = fi
        if np.array_equal(original, hashed_img):
            break

    return hashed_img
