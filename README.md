# Stochastic Simulation

Assignments for the course Stochastic Simulation, University of Amsterdam (23-24).

Solely intended for educational purposes.

<br/>

# Assignment 1

See notebook `Notebook.ipynb` for my submission of assignment 1. This notebook uses the modules in `mandelbrot` package, which adheres to the following structure:

```
mandelbrot/
│
├── sampling/
│   ├── __init__.py
│   ├── convergence.py          # Handles convergence analyses
│   ├── latin_hypercube.py      # Latin hypercube sampling
│   ├── monte_carlo.py          # Monte Carlo sampling (+ improvements)
│   ├── orthogonal.py           # Orthogonal sampling
│   └── pure_random.py          # Pure random sampling
│
├── utils/
│   ├── __init__.py
│   └── utils.py                # Generic utility functions
│
├── __init__.py
├── main.py                     # Orchastrates the computations
├── mandelbrot.py               # Handles Mandelbrot set
├── plotter.py                  # Handles plotting
└── requirements.txt            # Project dependencies
```
