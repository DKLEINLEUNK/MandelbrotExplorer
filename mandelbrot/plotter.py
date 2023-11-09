import matplotlib.pyplot as plt

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