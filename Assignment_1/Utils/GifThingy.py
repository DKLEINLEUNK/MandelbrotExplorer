import numpy as np
import matplotlib.pyplot as plt
import imageio

# Define the size of the image (pixels in x and y)
width, height = 600, 400

# Generate a list of complex coordinates (these are the points in the plane we'll check)
x = np.linspace(-2.5, 1.5, width)
y = np.linspace(-1, 1, height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Define the number of iterations we'll use to determine if a point is in the set
max_iter = 256

# Define the number of frames for the zoom
frames = 120
zoom_factor = 1.03

# Define a function that checks if a point is in the Mandelbrot set
def mandelbrot(c, max_iter):
    z = c
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

# Create a list to store the frames
images = []

# Generate each frame
for i in range(frames):
    # Print progress
    print(f"Rendering frame {i+1}/{frames}...")
    # Generate the Mandelbrot set
    mandel = np.array([[mandelbrot(c, max_iter) for c in row] for row in Z])
    # Normalize the image
    mandel = mandel / mandel.max()
    # Create the image
    plt.imshow(mandel, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap='hot')
    plt.axis('off')
    # Grab the image, and append to the list of images
    plt.savefig(f"frame_{i:03}.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    images.append(imageio.imread(f"frame_{i:03}.png"))
    # Adjust the bounds to zoom
    zoom = zoom_factor ** i
    X, Y = np.meshgrid(np.linspace(-2.5/zoom, 1.5/zoom, width), np.linspace(-1/zoom, 1/zoom, height))
    Z = X + 1j * Y

# Save the frames as an animated GIF
print("Creating GIF...")
imageio.mimsave('mandelbrot_zoom.gif', images, fps=12)

# Remove the individual frames (optional)
import os
for i in range(frames):
    os.remove(f"frame_{i:03}.png")

print("GIF saved as 'mandelbrot_zoom.gif'.")
