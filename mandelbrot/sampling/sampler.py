import numpy as np


def in_mandelbrot(c, MAX_ITER=256):
    '''Determines whether a complex number is in the Mandelbrot set.
    
    :param c: complex number
    '''
    z = 0                   # current value of z
    n_iter = 0              # current number of iterations

    while abs(z) <= 2 and n_iter < MAX_ITER:
        z = z**2 + c        # execute z = z^2 + c
        n_iter += 1         # increment the number of iterations

    if n_iter == MAX_ITER:
        return False  # as c did not result in a bounded sequence
    else:
        return True


def monte_carlo_sampling(grid: np.ndarray, n_samples=1000, max_iter=256):
    '''Performs Monte Carlo sampling on a grid of complex numbers.

    :param grid: grid of complex numbers
    :type grid: numpy.ndarray

    :param n_samples: number of samples to take
    :type n_samples: int

    :return: number of points in the sample that lie within the Mandelbrot set
    '''
    print(f'Grid size: {grid.size}')

    flat_grid = grid.flatten()  # 1D array of complex numbers
    print(f'Flattened grid size: {flat_grid.size}')

    # Select n_samples indices at random from the grid
    indices = np.random.choice(flat_grid.size, size=n_samples, replace=False)
    sample = flat_grid[indices]
    print(f'Sample size: {sample.size}')

    # Iterate each point in the sample, determine whether it lies within the Mandelbrot set
    is_in_mandelbrot = np.zeros(n_samples, dtype=bool)
    for i in range(len(sample)):
        is_in_mandelbrot[i] = in_mandelbrot(sample[i], MAX_ITER=max_iter)

    print(f'\n In Mandelbrot: {is_in_mandelbrot} \n')

    print(f'Number of points in the sample that lie within the Mandelbrot set: {np.sum(is_in_mandelbrot)}')

    return np.sum(is_in_mandelbrot)

def pure_random_sampling():
    pass


def latin_hypercube_sampling():
    pass


def orthogonal_array_sampling():
    pass
