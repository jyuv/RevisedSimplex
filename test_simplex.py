import random
import numpy as np
from scipy.optimize import linprog
from Simplex import Simplex, ResultCode

RANDOM_CASES = 100
MAX_M = 15
MAX_N = 10

SIMPLEX_SOLVER = Simplex()


def generate_random_case():
    m, n = [random.randint(1, end) for end in (MAX_M, MAX_N)]
    A = np.random.randint(-2147483648, 2147483647, size=(m, n))
    b = np.random.randint(-2147483648, 2147483647, size=m)
    c = np.random.randint(-2147483648, 2147483647, size=n)

    return A, b, c


def test_unbounded():
    A = np.array([[-1, 1], [-2, -2], [-1, 4]])
    b = np.array([-1, -6, 2])
    c = np.array([1, 3])

    assert SIMPLEX_SOLVER.has_feasible_solution(A, b, c)
    assert SIMPLEX_SOLVER.get_optimal_solution(A, b, c)[0] == \
           ResultCode.UNBOUNDED_OPTIMAL


def test_infeasible():
    A = np.array([[1], [-1]])
    b = np.array([5, -6])
    c = np.array([3])

    assert not SIMPLEX_SOLVER.has_feasible_solution(A, b, c)
    assert SIMPLEX_SOLVER.get_optimal_solution(A, b, c)[0] == \
           ResultCode.INFEASIBLE


def test_finite_feasible():
    # optimal solution is 10.5
    A = np.array([[1, 1, 2], [2, 0, 3], [2, 1, 3]])
    b = np.array([4, 5, 7])
    c = np.array([3, 2, 4])

    assert SIMPLEX_SOLVER.has_feasible_solution(A, b, c)
    optimal_res = SIMPLEX_SOLVER.get_optimal_solution(A, b, c)
    assert optimal_res[0] == ResultCode.FINITE_OPTIMAL
    assert optimal_res[2] == 10.5


def test_random_cases():
    """
    for case in range(RANDOM_CASES):
        A, b, c = generate_random_case()
        print(A, b, c)
        linprog(A_ub=A, b_ub=b, c=c, method="revised simplex")
    """
    return NotImplementedError


