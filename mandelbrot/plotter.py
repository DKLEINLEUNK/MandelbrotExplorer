import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd

from .utils import estimate_mean_area


def plot_mandelbrot_set(mandelbrot_set):
    '''Plots the Mandelbrot set.
    
    :param mandelbrot_set: the Mandelbrot set to plot
    '''
    # check if the set has been generated
    if not mandelbrot_set.is_set:
        raise Exception('Mandelbrot set has not been generated, please use `.set()` method.')
    
    h = plt.contourf(mandelbrot_set.x, mandelbrot_set.y, mandelbrot_set.grid)
    plt.axis('scaled')
    plt.colorbar()
    plt.show()


def plot_mandbrot_sample(points: ndarray, point_in_mandelbrot, plot_color):
    '''Plots a sample of complex numbers and indicates which points lie within the Mandelbrot set.
    
    :param points: an array of points
    :param point_in_mandelbrot: a boolean array indicating whether each point lies within the Mandelbrot set
    :param plot_color: the color of the points that lie within the Mandelbrot set
    '''
    plt.figure(figsize=(5, 5), dpi=900)
    plt.plot(points.real[point_in_mandelbrot], points.imag[point_in_mandelbrot], '.', color=plot_color, markersize=0.3)
    plt.show()


def plot_area_estimates(sampler_function, mandelbrot, n_means=50, max_iter=256):
    '''Plots the mean estimates of the area of the Mandelbrot set for different numbers of samples.
    
    :param sampler_function: function to perform the sampling
    :param mandelbrot: Mandelbrot set
    :param n_mean: number of means to estimate
    :param max_iter: maximum number of iterations to perform
    
    :return: estimated mean area of the Mandelbrot set
    '''
    n_samples = [10, 100, 1000, 10_000, 100_000, 1_000_000]
    output_len = len(n_samples)

    # if n_samples == None:
    #     n_samples = 10_000

    mean_areas = np.zeros(output_len)
    mean_stds = np.zeros(output_len)
    mean_errs = np.zeros(output_len)

    for sample_size in n_samples:
        
        mean_area, mean_std, mean_err = estimate_mean_area(
            repeats=n_means, 
            sampler_function=sampler_function,
            mandelbrot=mandelbrot, 
            n_samples=sample_size, 
            max_iter=max_iter
        )
        
        print(f'Area s={sample_size}: {mean_area} +/- {mean_err} (err)')
        
        # mean_areas[i], _, _ = estimate_mean_area(repeats=1, sampler_function=sampler_function, mandelbrot=mandelbrot, n_samples=n_samples, max_iter=max_iter)
        # mean, std, err = utils.estimate_mean_area(repeats=250, sampler_function=pr.sampler, mandelbrot=pr_mandelbrot, n_samples=1_000, max_iter=256)
    
    plt.figure(figsize=(5, 5))
    plt.hist(n_means, mean_areas)
    plt.show()


def plot_sampling_illustration(mandelbrot, sampler, latin_hypercube=False, orthogonal=False):
    '''Plots an illustration of input sampler.
    
    :param mandelbrot: Mandelbrot set
    :param sampler: sampler function
    :param orthogonal: whether `sampler` uses orthogonal sampling
    '''
    # 0. Get the name of the sampler:
    if latin_hypercube:
        sampler.__name__ = 'Latin Hypercube'
    elif orthogonal:
        sampler.__name__ = 'Orthogonal'
    else:
        sampler.__name__ = 'Pure Random'

    # 1. Initialize the plot layout
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # 2. Left plot: Small sample size, with strata
    # 2.1. Sample the Mandelbrot set:
    if orthogonal:
        n_samples = 100
        x_strata, y_strata, sample, in_mandelbrot = sampler(mandelbrot, n_samples, 256, return_lowerbounds=True)
    else:
        n_samples = 25
        sample, in_mandelbrot = sampler(mandelbrot, n_samples, 256)
        x_strata = np.linspace(mandelbrot.x_min, mandelbrot.x_max, num=n_samples + 1)
        y_strata = np.linspace(mandelbrot.y_min, mandelbrot.y_max, num=n_samples + 1)

    # 2.2. Plot the strata:
    for x in x_strata:
        axs[0].axvline(x, color='grey', linestyle='--', lw=0.5)
    for y in y_strata:
        axs[0].axhline(y, color='grey', linestyle='--', lw=0.5)

    # 2.3. Scatter the first sample:
    axs[0].scatter(sample.real, sample.imag, c=in_mandelbrot, cmap='viridis')
    axs[0].set_title(f"$s = {len(sample)}$")
    axs[0].set_xlabel("Real")
    axs[0].set_ylabel("Imaginary")

    # 3. Right plot: Large sample size, without strata
    sample2, in_mandelbrot2 = sampler(mandelbrot, 1_000, 256)
    axs[1].scatter(sample2.real, sample2.imag, c=in_mandelbrot2, cmap='viridis')
    axs[1].set_title("$s = 1,000$")
    axs[1].set_xlabel("Real")
    axs[1].set_yticks([])
    
    # 4. Show the plot:
    for ax in axs:
        ax.set_xlim([mandelbrot.x_min, mandelbrot.x_max])
        ax.set_ylim([mandelbrot.y_min, mandelbrot.y_max])
    fig.suptitle(f"{sampler.__name__} Sampling", fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_area_j_s(abbr='pr', sample_sizes=[100, 1_000, 10_000, 100_000, 1_000_000]):
    
    # 1. Initialize the plot layout & line styling
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    alphas = [0.4, 0.6, 0.8, 1, 0.8]
    plasma = plt.cm.get_cmap('plasma', len(sample_sizes))
    legend_names = [r'= 100', r'= 1,000', r'= 10,000', r'= 100,000', r'= 1,000,000']

    # 2. Plot the estimates
    for i in np.arange(len(sample_sizes)):
        areas_s = pd.read_csv(f'data/{abbr}_A_s_{sample_sizes[i]}.csv')
        
        # 2.1. Left plot:
        axs[0].plot(
            areas_s['n_iters'], 
            areas_s['area'],
            label=legend_names[i],
            alpha=alphas[i],
            color=plasma(i)
        )

        # 2.2. Right plot: Zoomed
        areas_s_zoomed = areas_s[(areas_s['n_iters'] >= 100) & (areas_s['n_iters'] <= 200)]
        axs[1].plot(
            areas_s_zoomed['n_iters'], 
            areas_s_zoomed['area'],
            alpha=alphas[i],
            color=plasma(i)
        )
        axs[1].set_ylim([0.8, 2.4])

    # 3. Show plot:
    fig.legend(
        loc='upper center', 
        framealpha=1,
        ncol=5, 
        fontsize=12,
        title=f'Sample Size, s',
        title_fontsize=16,
        columnspacing=1.5,
        handletextpad=0.5
    )
    fig.tight_layout()
    plt.show()
