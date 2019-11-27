import os
import cv2

# import numpy as np
import image_quantization as iq


def get_images(path):
    return [cv2.imread(path + f, -1) for f in os.listdir(path)]


def main():
    # expected dataset format: two folders: one "original", one "segmented".
    # names for both versions of each image should be the same, except
    # the ones in the segmented folder should have a "_mask" appended
    # to its name. filetypes Can be different.

    # dataset
    path_to_dataset = (
        "/home/annelaure/school/concordia_masters_comp_sci/"
        "COMP6731/project/Pattern_Project/dataset/example/"
    )
    # originals = get_images(path_to_dataset + "original/")
    # segmented = get_images(path_to_dataset + "segmented/")
    cow = cv2.imread(path_to_dataset + "original/cow_2008_000711.jpg")
    bins, reduced_image_bins = iq.get_colour_bins(cow, 3)
    reduced_image = iq.make_quantized_image(bins, reduced_image_bins)

    # example
    cv2.imshow("cow", reduced_image)
    cv2.waitKey(0)
    smoothed_bins = iq.smooth_image(bins, reduced_image_bins, 0.066)
    cv2.imshow("cow", iq.make_quantized_image(bins, smoothed_bins))
    cv2.waitKey(0)


main()
