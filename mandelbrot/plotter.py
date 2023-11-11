import matplotlib.pyplot as plt
from numpy import ndarray

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


def plot_mandbrot_sample(points: ndarray, point_in_mandelbrot):
    '''Plots a sample of complex numbers and indicates which points lie within the Mandelbrot set.
    
    :param points: an array of points
    :param point_in_mandelbrot: a boolean array indicating whether each point lies within the Mandelbrot set
    '''
    plt.plot(points.real[point_in_mandelbrot], points.imag[point_in_mandelbrot], 'b.')
    plt.show()    