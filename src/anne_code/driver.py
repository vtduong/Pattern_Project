import os
import cv2
import better_image_quantization as biq
import graph_cuts as gc


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
    print("fetching cow")
    cow = cv2.imread(path_to_dataset + "original/cow_2008_000711.jpg")
    cv2.imshow("cow", cow)

    print("hashing cow")
    hashed_cow = biq.hash_img(cow, 2, 255)
    mean_colours = biq.get_mean_colour_per_bin(cow, hashed_cow)
    reduced_cow = biq.make_img_with_assigned_colours(mean_colours, hashed_cow)
    cv2.imshow("hashed cow", reduced_cow)

    print("filtering low frequency colours")
    filtered_hashed_cow = biq.filter_low_frequency_colours(hashed_cow)
    mean_colours = biq.get_mean_colour_per_bin(cow, filtered_hashed_cow)
    filtered_cow = biq.make_img_with_assigned_colours(mean_colours, filtered_hashed_cow)
    cv2.imshow("filtered cow", filtered_cow)

    # print("fast alpha expansion")
    # segmented_hashed_cow = gc.img_segm(filtered_hashed_cow)
    # mean_colours = biq.get_mean_colour_per_bin(cow, segmented_hashed_cow)
    # segmented_cow = biq.make_img_with_assigned_colours(
    #    mean_colours, segmented_hashed_cow
    # )
    # cv2.imshow("segmented cow", reduced_cow)
    cv2.waitKey(0)

    ## smoothed_bins = iq.smooth_image(bins, reduced_image_bins, 0.066)

    # segmented_cow = gc.img_segm(hashed_cow)
    # mean_colours = biq.get_mean_colour_per_bin(cow, segmented_cow)
    # reduced_cow = biq.make_img_with_assigned_colours(mean_colours, segmented_cow)
    # cv2.imshow("segmented cow", reduced_cow)
    # cv2.waitKey(0)


main()
