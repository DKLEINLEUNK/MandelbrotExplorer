import matplotlib.pyplot as plt
import numpy as np

from mandelbrot.Mandelbrot import Mandelbrot
import mandelbrot.sampling.monte_carlo as mc

def main():
    # Generate the Mandelbrot set and store the complex grid
    mb = Mandelbrot(set=False)
    grid = mb.grid

    # Find the number of points in the Mandelbrot set using Monte Carlo sampling from the complex grid
    n_samples = 100_000
    mc_sample, mc_in_mandelbrot = mc.sampler(grid, n_samples, max_iter=256, verbose=False)

    # Estimate the area and plot the results
    area_estimate = mc_in_mandelbrot.size / np.sum(mc_in_mandelbrot)
    print(f'Area estimate: {area_estimate}')
    # area_estimate_2 = np.sum(mc_in_mandelbrot) / (mc_in_mandelbrot.size * (mandelbrot.x_max - mandelbrot.x_min) * (mandelbrot.y_max - mandelbrot.y_min))
    # print(f'Area estimate 2: {area_estimate_2}')
    plt.plot(mc_sample.real[mc_in_mandelbrot], mc_sample.imag[mc_in_mandelbrot], 'b.')
    plt.show()

if __name__ == '__main__':
    main()