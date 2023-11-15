import numpy as np

from ..Mandelbrot import Mandelbrot
from ..plotter import plot_mandbrot_sample
from ..utils import in_mandelbrot


def sampler(mandelbrot: Mandelbrot, n_samples=1000, max_iter=256, plot=False, verbose=False):
    '''Performs Latin hypercube sampling on a grid of complex numbers.'''
    # Goal: Sample one sample from each unique row and column of a grid
    n_samples = n_samples + 1  # Add 1 to account for the first sample

    # 1. Make `n_sample` strata of equal length along the real and imaginary axes
    x_interval = np.linspace(mandelbrot.x_min, mandelbrot.x_max, num=n_samples)
    x_strata = np.array(list(zip(x_interval[:-1], x_interval[1:])))
    y_interval = np.linspace(mandelbrot.y_min, mandelbrot.y_max, num=n_samples)
    y_strata = np.array(list(zip(y_interval[:-1], y_interval[1:])))
    
    if verbose:
        print(f'x_interval: {x_interval}')
        print(f'x_strata: {x_strata}')
        print(f'len(x_strata): {len(x_strata)}')

        print(f'y_interval: {y_interval}')
        print(f'y_strata: {y_strata}')
        print(f'len(y_strata): {len(y_strata)}')

    # 2. Generate random permutations of either row and column stratum pairs (TODO ensure that randomization of only rows or columns is valid)
    np.random.shuffle(y_strata)
    strata = np.array(list(zip(x_strata, y_strata)))
    
    if verbose:
        print(f'Permutations: {strata}, with length {strata.size}')

    # 3. Sample one point from each permutation (randomly between the strata's bounds)
    sample = np.array([])
    for stratum in strata:
        x_bounds = stratum[0]
        y_bounds = stratum[1]
        x = np.random.uniform(x_bounds[0], x_bounds[1])
        y = np.random.uniform(y_bounds[0], y_bounds[1])        
        sample = np.append(sample, x + 1j * y)
        
        if verbose:
            print(f'sampled {x + 1j * y}')
    
    if verbose:
        print(f'Number of points in the sample: {len(sample)}')
        print(f'Last 5 samples: {sample[-5:]}')

    # 4. Iterate each point in the sample, determine whether it lies within the Mandelbrot set
    is_in_mandelbrot = np.zeros(sample.size, dtype=bool)
    for i in range(sample.size):
        is_in_mandelbrot[i] = in_mandelbrot(sample[i], MAX_ITER=max_iter)
        
        if (verbose):
            print(f'point: {sample[i]}, in_mandelbrot: {is_in_mandelbrot[i]}')

    if verbose:
        print(f'Number of points in the sample that lie within the Mandelbrot set: {np.sum(is_in_mandelbrot)}')
        print(f'Number of points in the sample that do not lie within the Mandelbrot set: {np.sum(~is_in_mandelbrot)}')
        print(f'Number of points in the boolean: {len(is_in_mandelbrot)}')
        print(f'Last 5 booleans: {is_in_mandelbrot[-5:]}')

    if plot:
        plot_mandbrot_sample(sample, is_in_mandelbrot, plot_color='#9370DB')

    return sample, is_in_mandelbrot
