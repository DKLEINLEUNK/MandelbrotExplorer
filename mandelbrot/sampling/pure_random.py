import numpy as np

from ..Mandelbrot import Mandelbrot
from ..plotter import plot_mandbrot_sample
from ..utils import in_mandelbrot_vectorized


def sampler(mandelbrot: Mandelbrot, n_samples=1000, max_iter=256, plot=False, verbose=False):
    '''Performs Monte Carlo sampling on a grid of complex numbers.
    
    Now with optimization.

    :param mandelbrot: Mandelbrot set
    :param n_samples: number of samples to take
    :param max_iter: maximum number of iterations to perform
    :param plot: whether to plot the sample
    :param verbose: whether to print out information about the sampling process

    :return: a sample of complex numbers and a boolean array indicating whether each point in the sample lies within the Mandelbrot set
    '''
    # 1. Randomly select points on the x and y axes:
    x_samples = np.random.uniform(mandelbrot.x_min, mandelbrot.x_max, size=n_samples)
    y_samples = np.random.uniform(mandelbrot.y_min, mandelbrot.y_max, size=n_samples)

    # 2. Combine the x and y samples to get the sample points:
    sample = x_samples + 1j * y_samples

    # 3. Check whether the sampled points are in the set or not:
    is_in_mandelbrot = in_mandelbrot_vectorized(sample, MAX_ITER=max_iter)

    if verbose:
        print(f'x_samples: {x_samples}')
        print(f'y_samples: {y_samples}')
        print(f'sample: {sample}')
        print(f'is_in_mandelbrot: {is_in_mandelbrot}')
    
    if plot:
        plot_mandbrot_sample(sample, is_in_mandelbrot, plot_color='#5f3cff')
    
    return sample, is_in_mandelbrot


def old_sampler(mandelbrot: Mandelbrot, n_samples=1000, max_iter=256, plot=False, verbose=False):
    '''Performs Monte Carlo sampling on a grid of complex numbers.

    :param mandelbrot: Mandelbrot set
    :param n_samples: number of samples to take
    :param max_iter: maximum number of iterations to perform
    :param plot: whether to plot the sample
    :param verbose: whether to print out information about the sampling process

    :return: a sample of complex numbers and a boolean array indicating whether each point in the sample lies within the Mandelbrot set
    '''
    # 1. Flatten the grid of the mandelbrot set:
    flat_grid = mandelbrot.grid.flatten()

    # 2. Randomly select n_samples indices from the grid:
    indices = np.random.choice(flat_grid.size, size=n_samples, replace=False)
    sample = flat_grid[indices]

    # 3. Determine whether the samples lie within the Mandelbrot set:
    is_in_mandelbrot = in_mandelbrot_vectorized(sample, MAX_ITER=max_iter)

    if verbose:
        print(f'Grid size: {mandelbrot.grid.size}')
        print(f'Flattened grid size: {flat_grid.size}')
        print(f'#Indices: {indices.size}')
        print(f'Indices: {indices}')
        print(f'Sample size: {sample.size}')
        print(f'\nNumber of points in the sample that lie within the Mandelbrot set: {np.sum(is_in_mandelbrot)}')

    if plot:
        plot_mandbrot_sample(sample, is_in_mandelbrot, plot_color='b')

    return sample, is_in_mandelbrot
