import os
import logging
import time

import numpy as np
import pandas as pd

from .Mandelbrot import Mandelbrot



#---------------------------
# PROJECT UTILITIES
#---------------------------

# Check if a complex number is in the Mandelbrot set
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
        return True
    else:
        return False  # as c did not result in a bounded sequence


def in_mandelbrot_vectorized(samples: np.ndarray, MAX_ITER=256):
    '''Determines whether a complex number is in the Mandelbrot set, but then this time with sick performance gainz ;)
    
    :param samples: complex numbers
    '''
    z = np.zeros_like(samples)
    to_do = np.ones(samples.shape, dtype=bool)  # array of all samples still being processed
    in_mandelbrot = np.full(samples.shape, True, dtype=bool)
    
    # 1. Use numpy's array broadcasting to update all samples at once (instead of looping over each sample) 
    for i in range(MAX_ITER):
        
        # 1.1 Update samples still being processed
        z[to_do] = z[to_do]**2 + samples[to_do]

        # 1.2 Update to_do array given the updated |z| values
        to_do &= (np.abs(z) <= 2)
        in_mandelbrot &= to_do

    return in_mandelbrot


def in_mandelbrot_vectorized_count(Mandelbrot: Mandelbrot, max_iter):
    C = Mandelbrot.xy_grid
    z = np.zeros_like(C)
    to_do = np.ones(C.shape, dtype=bool)  # array of all samples still being processed
    count = np.zeros(C.shape, dtype=int)
    
    # 1. Use numpy's array broadcasting to update all samples at once (instead of looping over each sample) 
    for i in range(max_iter):
        
        # 1.1 Update samples (with count) still being processed
        z[to_do] = z[to_do]**2 + C[to_do]
        count[to_do] = i
        
        # 1.2 Update to_do array given the updated |z| values
        to_do &= (np.abs(z) <= 2)

    return count


def estimate_mandelbrot_area(sample: np.ndarray, in_mandelbrot: np.ndarray, mandelbrot, verbose=False):
    '''Estimates the area of the Mandelbrot set within a given range.
    
    :param samples: sample of complex numbers
    :param in_mandelbrot: boolean array indicating whether each sample is in the Mandelbrot set
    :param mandelbrot: Mandelbrot set

    :return: estimated area of the Mandelbrot set
    '''
    proportion_in_set = np.sum(in_mandelbrot) / sample.size
    
    area_grid = (mandelbrot.x_max - mandelbrot.x_min) * (mandelbrot.y_max - mandelbrot.y_min)

    # Estimate the area of the Mandelbrot set
    area = proportion_in_set * area_grid

    if verbose:
        print(f'A = {area} for {sample.size} samples & 256 iterations')

    return area


def estimate_mean_area(repeats: int, sampler_function, mandelbrot, n_samples=10_000, n_iters=256, verbose = False):
    '''Estimates the mean of a sample of complex numbers.
    
    :param repeats: number of times to repeat the sampling process
    :param sampler: function to perform the sampling
    :param mandelbrot: Mandelbrot set
    :param verbose: whether to print out information about the sampling process

    :return: estimated mean area of the Mandelbrot set
    '''
    if verbose:
        print(f'Estimating mean area for {repeats} repeats...')

    # 1. Estimate the mean area of the Mandelbrot set for `repeats` samples:
    areas = np.zeros(repeats)
    for i in np.arange(repeats):

        # 1.1 Use the sampler function:
        sample, is_in_mandelbrot = sampler_function(mandelbrot, n_samples, n_iters)

        # 1.2 Estimate the area of the Mandelbrot set:
        area = estimate_mandelbrot_area(sample, is_in_mandelbrot, mandelbrot)
        areas[i] = area

        if verbose:
            print(f'Area {i}: {area}')

    # 2. Calculate the mean area (& standard deviation and error):
    mean_area = np.mean(areas)
    mean_area_std = np.std(areas)
    mean_area_err = mean_area_std / np.sqrt(repeats)

    return mean_area, mean_area_std, mean_area_err


def estimate_A_j_s(mandelbrot, sampler, abbr='tmp', J=np.arange(1, 257, 2), S=np.array([100, 1_000, 10_000, 100_000, 1_000_000])):
    n_iters = J
    n_samples = S
    
    for s in S:

        print(f'Running estimations for s = {s}...')
        
        areas = np.zeros(len(n_iters))

        for i in n_iters:

            sample, is_in_mandelbrot = sampler(mandelbrot, s, i)
            area = estimate_mandelbrot_area(sample, is_in_mandelbrot, mandelbrot)
            areas[i//2] = area

        print('Done!')
        
        pd.DataFrame({
            'n_iters': n_iters,
            'area': areas
        }).to_csv(f'data/{abbr}_A_s_{s}.csv', index=False)


#---------------------------
# GENERIC LIBRARY UTILITIES
#---------------------------

# File and Directory Management: Functions to create directories for saving plots, check if a file already exists, or handle file paths.
def ensure_directory_exists(path):
    """Ensure that a directory exists, and if not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)


# Data Handling: Functions to handle the input data (i.e. data processing) or parameters (e.g. data validation) to ensure they meet certain criteria.
def normalize_data(data):
    """Normalize data to a certain range or format."""
    # Implementation depends on the data format and requirements
    ...


def validate_positive_integer(value, name="Value"):
    """Ensure a value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer. Got {value}.")


# Formatting and Conversions: Functions to format data for plotting or convert between different data types or structures.
def format_plot_title(title):
    """Format a title string for a plot."""
    return title.replace("_", " ").title()


# Logging and Debugging: Functions to log information, warnings, or errors, or to assist with debugging.
def configure_logging(level=logging.INFO):
    """Configure logging for the project."""
    logging.basicConfig(level=level)


def timing_decorator(func):
    """A decorator to measure the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} executed in {end_time - start_time:.2f}s")
        return result
    return wrapper


def handle_error(e):
    """Handle an error, use standardized logging."""
    logging.error(f"An error occurred: {str(e)}")
