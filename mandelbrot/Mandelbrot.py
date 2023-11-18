import numpy as np


class Mandelbrot:
    '''Class for generating the Mandelbrot set.'''

    def __init__(self, width=1000, height=1000, x_min=-1.7, x_max=0.7, y_min=-1.2, y_max=1.2, set=False):
        '''Generates a grid of complex numbers in the complex plane.

        :param width: number of points in the real part
        :param height: number of points in the imaginary part

        :param x_min: minimum value of the real part
        :param x_max: maximum value of the real part
        :param y_min: minimum value of the imaginary part
        :param y_max: maximum value of the imaginary part
        
        :param sparse: if True, returns sparse coordinate arrays

        :return: a grid of complex numbers
        '''

        self.width = width
        self.height = height  # TODO remove
        
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.x = np.linspace(x_min, x_max, num=width)
        self.y = np.linspace(y_min, y_max, num=height)

        self.x_grid, self.y_grid, self.xy_grid = self.generate_grid()
        

    def generate_grid(self):
        '''Generates a grid of complex numbers in the complex plane.'''
        xx, yy = np.meshgrid(self.x, self.y)
        zz = xx + 1j * yy  # complex grid

        return xx, yy, zz