import numpy as np
import better_image_quantization as iq


max_value = 10
channels = 4
number_of_buckets = 4
number_of_pixels = 5
# p = np.random.randint(0, max_value + 1, (number_of_pixels, channels))
# print("p", p)
# print("bins\n", iq.get_bins(channels, number_of_buckets, max_value))
# print("hashed image", iq.hash_image(p, number_of_buckets, max_value))

print(
    iq.get_bin(6, channels, number_of_buckets, max_value),
    iq.get_nearby_bins(
        iq.get_bin(6, channels, number_of_buckets, max_value),
        iq.get_bin_lims(channels, max_value),
    ),
)
