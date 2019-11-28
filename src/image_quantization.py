import numpy as np
import itertools

COLOR_MAX = 255  # assumes 16-bit RGB colour space


def retarded_range_helper(j, max_vals):
    # given a 2d array and max vals, returns acceptable python coords
    # for range around the j value
    vals = [[j[0] - 1, j[0] + 1], [j[1] - 1, j[1] + 1]]
    for i, j_val in enumerate(j):
        if j_val == 0:
            vals[i][0] = 0
        if j_val == max_vals[i] - 1:
            vals[i][1] = max_vals[i]
    return vals


def sort_by_rows(arr):
    return arr[np.lexsort(arr.T[::-1])]


def get_bins_index(orig, bins, col, max):
    # this is super inefficient
    for c in range(len(col)):
        # print(c)
        for i in range(len(bins)):
            if col[c] < bins[i][c] or bins[i][c] == COLOR_MAX:
                # print("PICKING", col[c], bins[i][c])
                bins = bins[bins[:, c] == bins[i][c]]
                break
    # return index of bin
    return np.where((orig == bins[0]).all(axis=1))[0][0]


def get_bins(spacing=None, num_of_seps_per_channel=None):
    # gets you the segmented bins for a specific spacing of each colour channel

    # different way of using this func, by not passing a spacing
    if num_of_seps_per_channel is not None:
        spacing = np.linspace(0, COLOR_MAX, num_of_seps_per_channel + 2)
    if spacing is None:
        print(
            "you must pass me at least a spacing param or a"
            "num_of_seps_per_channel param!"
        )
        return None

    return sort_by_rows(
        np.array(list(itertools.combinations_with_replacement(spacing, 3)))
    )


def get_colour_bins(img, num_of_seps_per_channel=3):
    spacing = np.linspace(0, COLOR_MAX, num_of_seps_per_channel + 1)[1:]
    bins = get_bins(spacing)
    orig_shape = img.shape  # to redr
    img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

    # get bin indexes for each pixel
    pixel_indexes = np.array(
        [(*p, get_bins_index(bins, bins, p, COLOR_MAX)) for p in img]
    )

    # get mean colour for each bin
    mean_colour = []
    for i in np.unique(pixel_indexes[:, 3]):
        pixels = pixel_indexes[pixel_indexes[:, 3] == i]
        mean_colour.append((len(pixels), np.mean(pixels[:, 0:3], axis=0).astype(int)))

    # returns the bin mean colours and how many pixels are each in shape,
    # and which bin each pixel got binned in
    return mean_colour, pixel_indexes[:, 3].reshape(orig_shape[:2])


def make_quantized_image(bins, image_bins, title="default"):
    # if you want to show the image according to the color bins
    # use this. pass the output of get_colour_bins to it.
    image = np.ones((*image_bins.shape, 3), dtype="uint8")
    for i in range(len(bins)):
        image[image_bins[:] == i] = np.array(bins[i][1])
    return image


def smooth_image(bins, image_bins, threshhold=0.66):
    print(image_bins.shape)
    total = sum(np.array(bins)[:, 0])
    for i, c in enumerate(bins):
        # if the frequency of this color is low, filter it out to nearby colors
        if c[0] / total < threshhold:

            print(i, c[0] / total, c, len(bins))
            indexes = np.argwhere(image_bins == i)
            for j in indexes:
                r = retarded_range_helper(j, image_bins.shape)
                image_bins[j[0], j[1]] = np.bincount(
                    image_bins[r[0][0] : r[0][1], r[1][0] : r[1][1]][0]
                ).argmax()
    return image_bins
