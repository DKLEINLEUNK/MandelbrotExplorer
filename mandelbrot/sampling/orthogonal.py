import itertools

import numpy as np

from ..Mandelbrot import Mandelbrot
from ..plotter import plot_mandbrot_sample
from ..utils import in_mandelbrot_vectorized


def sampler(mandelbrot: Mandelbrot, n_sample=1000, max_iter=256, plot=False, verbose=False, return_lowerbounds=False, use_for_region_of_interest=False):
    # ------------------------------------------------------------------
    # Sample the Mandelbrot set using orthogonal sampling and return the 
    # sampled points with boolean values indicating whether the point is 
    # in the set or not.
    # ------------------------------------------------------------------
    interval_length = round((np.sqrt(n_sample) + 1))

    # 1. Initialize the intervals:
    x_interval = np.linspace(mandelbrot.x_min, mandelbrot.x_max, num=interval_length)
    y_interval = np.linspace(mandelbrot.y_min, mandelbrot.y_max, num=interval_length)

    # 2. Calculate stratum width and height:
    x_width = (mandelbrot.x_max - mandelbrot.x_min) / interval_length
    y_height = (mandelbrot.y_max - mandelbrot.y_min) / interval_length

    # 3. Initialize vertical and horizontal grid with all lower bounds of each stratum:
    x_lower_bounds = np.repeat(x_interval[:-1], (interval_length-1))  # note, interval_length-1 because we don't want to include the upper bound to avoid duplicates
    y_lower_bounds = np.tile(y_interval[:-1], (interval_length-1))
    
    # 4. Form random offsets for each stratum:
    x_offsets = np.random.uniform(0, x_width, size=x_lower_bounds.shape)
    y_offsets = np.random.uniform(0, y_height, size=y_lower_bounds.shape)

    # 5. Add the offsets to the lower bounds to get the sampled points:
    sample = (x_lower_bounds + x_offsets) + 1j * (y_lower_bounds + y_offsets)

    # 6. Check whether the sampled points are in the set or not:
    is_in_mandelbrot = in_mandelbrot_vectorized(sample, MAX_ITER=max_iter)

    if verbose:
        print(f'x_width: {x_width}')
        print(f'y_height: {y_height}')
        print(f'\nlower bounds, x: {x_lower_bounds}')
        print(f'lower bounds, y: {y_lower_bounds}')
        print(f'lower bounds, x.shape: {x_lower_bounds.shape}')
        print(f'lower bounds, y.shape: {y_lower_bounds.shape}')
        print(f'\nx_offsets: {x_offsets}')
        print(f'y_offsets: {y_offsets}')
        print(f'\nsample: {sample}')
        print(f'sample.shape: {sample.shape}')
        print(f'\nis_in_mandelbrot: {is_in_mandelbrot}')

    if plot:
        plot_mandbrot_sample(sample, is_in_mandelbrot, plot_color='#C50974')

    if return_lowerbounds:
        return x_lower_bounds, y_lower_bounds, sample, is_in_mandelbrot

    if use_for_region_of_interest:
        return interval_length, x_interval, y_interval, x_lower_bounds, y_lower_bounds, sample, is_in_mandelbrot

    return sample, is_in_mandelbrot


def identify_area_of_interest(mandelbrot: Mandelbrot, n_sample=1000, max_iter=256):

    # 1. Ugly code, but hey, it works... Get the relevant info from sampler():
    interval_length, x_interval, y_interval, x_lower_bounds, y_lower_bounds, sample, is_in_mandelbrot = sampler(mandelbrot, n_sample, max_iter, use_for_region_of_interest=True)
    
    # 2. Initialize the area of interest:
    area_of_interest = np.zeros((interval_length - 1, interval_length - 1), dtype=bool)

    # 3. Inefficient, but again, it works... Iterate over all strata:
    for i in range(interval_length - 1):
        for j in range(interval_length - 1):
            
            # 3.1. Per stratum, get neighbors (right, bottom, & bottom-right):
            try:
                x_lbound = x_interval[i]
                x_ubound = x_interval[i + 2]
                y_lbound = y_interval[j]
                y_ubound = y_interval[j + 2]
            except IndexError:
                # print(f'Encountered IndexError at (i,j)=({i},{j}), skipping...')
                # I don't have time for any clever solutions...
                continue

            # 3.2. Check if stratum + neighbors contain in_mandelbrot AND not in_mandelbrot points:
            if np.any(is_in_mandelbrot[
                (x_lower_bounds >= x_lbound) & (x_lower_bounds < x_ubound) &
                (y_lower_bounds >= y_lbound) & (y_lower_bounds < y_ubound)
            ]) and np.any(~is_in_mandelbrot[
                (x_lower_bounds >= x_lbound) & (x_lower_bounds < x_ubound) &
                (y_lower_bounds >= y_lbound) & (y_lower_bounds < y_ubound)
            ]):
                # 3.2.1. If so, current stratum is (hopefully) near the boundary region:
                area_of_interest[j, i] = True
        
    return interval_length, x_interval, y_interval, area_of_interest, sample, is_in_mandelbrot


def draw_additional_samples(n_samples, interval_length, x_interval, y_interval, area_of_interest):

    # 1. Initialize the additional samples:
    samples = []

    # 2. Cycle through the area of interest:
    for i in range(interval_length - 1):
        for j in range(interval_length - 1):
            if area_of_interest[i, j]:

                # 2.1 Locate stratum:
                x_lbound, x_ubound = x_interval[i], x_interval[i + 1]
                y_lbound, y_ubound = y_interval[j], y_interval[j + 1]

                # 2.2 Generate samples in the stratum:
                x_new_samples = np.random.uniform(x_lbound, x_ubound, size=n_samples)
                y_new_samples = np.random.uniform(y_lbound, y_ubound, size=n_samples)
                samples.extend(x_new_samples + 1j * y_new_samples)
    
    return samples


def optimized_sampler():
    # new_in_mandelbrot = in_mandelbrot_vectorized(new_samples, MAX_ITER=max_iter)
    pass
