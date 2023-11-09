def mandelbrot(c, max_iter):
    z = c
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n
