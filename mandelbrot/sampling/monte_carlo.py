import numpy as np

from ..utils import in_mandelbrot

def sampler(grid: np.ndarray, n_samples=1000, max_iter=256, verbose=False):
    '''Performs Monte Carlo sampling on a grid of complex numbers.

    :param grid: grid of complex numbers
    :type grid: numpy.ndarray

    :param n_samples: number of samples to take
    :type n_samples: int

    :return: a sample of complex numbers and a boolean array indicating whether each point in the sample lies within the Mandelbrot set
    '''
    flat_grid = grid.flatten()  # 1D array of complex numbers

    # Select n_samples indices at random from the grid
    indices = np.random.choice(flat_grid.size, size=n_samples, replace=False)
    sample = flat_grid[indices]

    # Iterate each point in the sample, determine whether it lies within the Mandelbrot set
    is_in_mandelbrot = np.zeros(n_samples, dtype=bool)
    for i in range(len(sample)):
        is_in_mandelbrot[i] = in_mandelbrot(sample[i], MAX_ITER=max_iter)

    if verbose:
        print(f'Grid size: {grid.size}')
        print(f'Flattened grid size: {flat_grid.size}')
        print(f'#Indices: {indices.size}')
        print(f'Indices: {indices}')
        print(f'Sample size: {sample.size}')
        print(f'\nNumber of points in the sample that lie within the Mandelbrot set: {np.sum(is_in_mandelbrot)}')

    return sample, is_in_mandelbrot
