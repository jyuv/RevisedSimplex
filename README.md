# RevisedSimplex

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Overview
A numpy based implementation of revised simplex with numerical stability checks. The implementation was found **16% faster than scipy's revised-simplex** method running a series of random configurations. 

## The Problem - Linear Programming Maximization Problem
This repositry offers an implementation for an algorithm that solves Linear Programming configurations. A Linear Programming Maximization Problem is defined as:

<p align=center>
<img src="https://user-images.githubusercontent.com/45140931/120074925-d628c680-c0a7-11eb-9deb-1802cb1ca76b.png">
</p>


Where the A matrix and b & c vectors are given and we need to find a valid x vector that maximizes the score function.

## The Algorithm - Revised Simplex
The Revised Simplex method is a variant of the classic simplex algorithm for solving Linear Programming configurations. This method allows for a greater computational efficiency. For more details you can watch the slides [here](https://imada.sdu.dk/~marco/Teaching/AY2016-2017/DM559/Slides/dm545-lec7.pdf).

## The Implementation & How To Use
The implementation is based on numpy. It includes  a number of numerical stability checks utilizing eta matrices and lu factorization to keep it efficient.
### How To Use
The implementation allows the use of both Bland & Dantzig rules. The module provides two main external API functions:
1. **get_optimal_solution** - searches for an optimal solution. This function returns a SimplexResult object:
```python
class ResultCode(Enum):
    INFEASIBLE = 1
    FINITE_OPTIMAL = 2
    UNBOUNDED_OPTIMAL = 3
    CYCLE_DETECTED = 4


@dataclass
class SimplexResult(object):
    res_code: ResultCode
    assignment: Union[None, np.ndarray]
    optimal_score: Union[None, float]
```

2. **has_feasible_solution** - checks if there is a feasible solution for the configuration. Returns True\False.


Here is an example of how to use the module:

```python
import numpy as np
from Simplex import Simplex, PickingRule

solver = Simplex()
A = np.array([[-1, 1], [-2, -2], [-1, 4]])
b = np.array([-1, -6, 2])
c = np.array([1, 3])
rule = PickingRule.DANTZIG_RULE  # Or PickingRule.BLAND_RULE

# optimal using the default rule - bland's
solver.get_optimal_solution(A, b, c)

# optimal using a specified rule - dantzig's
solver.get_optimal_solution(A, b, c)

# check for existance of feasible solution
solver.has_feasible_solution(A, b, c)
```

## Performance Analysis
The performance of this repository was compared to scipy's revised-simplex method by running 10,000 random generated configurations.
The results show **an improvement of 16%** compared to scipy's performance when **running the full revised simplex** (searching for optimal solution).
The more humble method **(has_feasible_solution) was 41% faster** than scipy's full revised simplex.
