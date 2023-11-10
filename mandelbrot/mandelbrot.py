import numpy as np

from plotter import plot_mandelbrot_set


class Mandelbrot:
    '''Class for generating the Mandelbrot set.'''

    def __init__(self, width=1000, height=1000, x_min=-1.7, x_max=0.7, y_min=-1.2, y_max=1.2, set=False):
        '''Generates a grid of complex numbers in the complex plane.

        :param width: number of points in the real part
        :type width: int

        :param height: number of points in the imaginary part
        :type height: int
        
        :param x_min: minimum value of the real part
        :type x_min: float

        :param x_max: maximum value of the real part
        :type x_max: float

        :param y_min: minimum value of the imaginary part
        :type y_min: float

        :param y_max: maximum value of the imaginary part
        :type y_max: float

        :param sparse: if True, returns sparse coordinate arrays
        :type sparse: bool

        :return: a grid of complex numbers
        '''

        self.width = width
        self.height = height
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.x = np.linspace(x_min, x_max, num=width)
        self.y = np.linspace(y_min, y_max, num=height)

        self.grid = self.generate_grid()

        self.is_set = False

        if set:
            self.set()


    def generate_grid(self):
        '''Generates a grid of complex numbers in the complex plane.'''
        xx, yy = np.meshgrid(self.x, self.y)
        zz = xx + 1j * yy  # complex grid

        return zz


    def set(self, max_iter=256, plot=False):
        '''Generates the Mandelbrot set.
        
        :param max_iter: maximum number of iterations
        :type max_iter: int

        :param plot: if True, plots the Mandelbrot set
        :type plot: bool
        '''
        MAX_ITER = max_iter

        for y in range(self.height):
        
            for x in range(self.width):
        
                c = self.grid[y][x]     # current complex number
                z = 0                   # current value of z
                n_iter = 0              # current number of iterations
                
                while abs(z) <= 2 and n_iter < MAX_ITER:
              
                    z = z**2 + c        # execute z = z^2 + c
                    n_iter += 1         # increment the number of iterations
                
                self.grid[y][x] = n_iter
        
        self.is_set = True

        if plot:
            plot_mandelbrot_set(self)


if __name__ == '__main__':
    from sampling import sampler

    mandelbrot = Mandelbrot(set=False)
    grid = mandelbrot.grid
    
    # Find the number of points in the Mandelbrot set using Monte Carlo sampling
    n_samples = 10_000
    points_in_mandelbrot = sampler.monte_carlo_sampling(grid, n_samples, max_iter=256)
    
    # area_estimate = points_in_mandelbrot / n_samples
    area_estimate = points_in_mandelbrot / n_samples * (mandelbrot.x_max - mandelbrot.x_min) * (mandelbrot.y_max - mandelbrot.y_min)

    print(f'Area estimate: {area_estimate}')


    # plot_mandelbrot_set(mandelbrot_set)
