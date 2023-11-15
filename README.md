# Stochastic Simulation

Assignments for the course Stochastic Simulation, University of Amsterdam (23-24).

Solely intended for educational purposes.

<br/>

# Assignment 1

See notebook `Notebook.ipynb` for our submission of assignment 1. This notebook uses the modules in `mandelbrot/`.

## TODO

### Large tasks
- [ ] Implement orthogonal sampling.
- [ ] Define a statistics generator for estimating integration accuracy.
- [ ] Compare how different quasi-random number algortithms perform.
- [ ] Test wether each sampling method adhered to all assumptions.
- [ ] Update all sampling algorithms with the optimized algorithm in `orthogonal.py`.

### Small tasks (additions and removals)
- [ ] In `orthogonal.py` replace `interval_length` with `n_samples` and infer interval length from that.
- [ ] Add a method to Mandelbrot class that calculates the area of the set.
- [ ] Add a calculation of the area: multiply points in set by total sample space.
- [ ] Remove `monte_carlo.py` module (and its traces in other modules).


## Project Outline
The `mandelbrot` package adheres to the following structure:

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
