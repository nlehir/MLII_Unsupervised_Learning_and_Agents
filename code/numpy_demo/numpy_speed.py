import numpy as np
from time import time
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

rng = np.random.default_rng()


def product_with_lists(list_1, list_2):
    print("sum of product with lists")
    result = 0
    for elem_1, elem_2 in zip(list_1, list_2):
        result += elem_1 * elem_2
    return result


def product_with_numpy(arr_1, arr_2):
    print("sum of products with numpy")
    element_wise_product = arr_1 * arr_2
    return element_wise_product.sum()


n_elements = int(1e7)

list_1 = [rng.random() for _ in range(n_elements)]
list_2 = [rng.random() for _ in range(n_elements)]

arr_1 = rng.random(n_elements)
arr_2 = rng.random(n_elements)

"""
with lists
"""

tic_list = time()
product_with_lists_ = product_with_lists(list_1, list_2)
toc_list = time()

"""
with numpy
"""

tic_arr = time()
product_with_numpy_ = product_with_numpy(arr_1, arr_2)
toc_arr = time()

print(f"time with lists: {toc_list-tic_list:.6f}")
print(f"time with numpy: {toc_arr-tic_arr:.6f}")

"""
save profiling
"""
profiler.disable()
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats("cumtime")
stats_data_file = "profile.prof"
stats.dump_stats(stats_data_file)
