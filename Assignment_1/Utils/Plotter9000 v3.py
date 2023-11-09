import numpy as np
import matplotlib.pyplot as plt

# Set the size of the plot
width, height = 800, 800

# Define the bounds of the plot in the complex plane
x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5

# Generate a grid of c-values
x, y = np.linspace(x_min, x_max, width), np.linspace(y_min, y_max, height)
C = np.meshgrid(x, y)

# Create a complex array of the c-values
complex_grid = C[0] + 1j * C[1]

# Initialize the array for the set
Mandelbrot_set = np.zeros((height, width))

# Define the maximum number of iterations
max_iter = 256

# Perform the iteration
for i in range(height):
    for j in range(width):
        c = complex_grid[i, j]
        z = 0
        for n in range(max_iter):
            z = z**2 + c
            if abs(z) > 2:
                Mandelbrot_set[i, j] = n
                break

# Normalize the set for better coloring
# Mandelbrot_set = Mandelbrot_set / Mandelbrot_set.max()

# Plot the Mandelbrot set
plt.imshow(Mandelbrot_set.T, extent=[x_min, x_max, y_min, y_max], cmap='hot')
plt.colorbar()
plt.title('Mandelbrot Set')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.show()
