import itertools

import numpy as np

from ..Mandelbrot import Mandelbrot
from ..plotter import plot_mandbrot_sample
from ..utils import in_mandelbrot_vectorized

def sampler(mandelbrot: Mandelbrot, interval_length=1000, max_iter=256, plot=False, verbose=False):
    # TODO: replace `interval_length` with `n_samples` and infer the interval length from that
    
    # ------------------------------------------------------------------
    # Sample the Mandelbrot set using orthogonal sampling and return the 
    # sampled points with boolean values indicating whether the point is 
    # in the set or not.
    # ------------------------------------------------------------------
    
    # 1. Initialize the intervals:
    x_interval = np.linspace(mandelbrot.x_min, mandelbrot.x_max, num=interval_length)
    y_interval = np.linspace(mandelbrot.y_min, mandelbrot.y_max, num=interval_length)

    # 2. Calculate stratum width and height:
    x_width = (mandelbrot.x_max - mandelbrot.x_min) / interval_length
    y_height = (mandelbrot.y_max - mandelbrot.y_min) / interval_length

    # 3. Initialize vertical and horizontal grid with all lower bounds of each stratum:
    x_lower_bounds = np.repeat(x_interval[:-1], interval_length)
    y_lower_bounds = np.tile(y_interval[:-1], interval_length)
    
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
        plot_mandbrot_sample(sample, is_in_mandelbrot, plot_color='#9370DB')
