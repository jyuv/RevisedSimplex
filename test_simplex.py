import math
import random

import numpy as np
from scipy.optimize import linprog
from Simplex import Simplex, ResultCode

RANDOM_CASES = 1000
MAX_M = 15
MAX_N = 10
MIN_INT_ELEMENT = -2147483648
MAX_INT_ELEMENT = 2147483647

LINPROG_RCODE_TO_RCODE = {
    0: ResultCode.FINITE_OPTIMAL,
    1: ResultCode.CYCLE_DETECTED,
    2: ResultCode.INFEASIBLE,
    3: ResultCode.UNBOUNDED_OPTIMAL,
}

SIMPLEX_SOLVER = Simplex()


def generate_random_case(
    max_m=MAX_M, max_n=MAX_N, min_int=MIN_INT_ELEMENT, max_int=MAX_INT_ELEMENT
):
    m, n = [random.randint(1, end) for end in (max_m, max_n)]
    A = np.random.randint(min_int, max_int, size=(m, n))
    b = np.random.randint(min_int, max_int, size=m)
    c = np.random.randint(min_int, max_int, size=n)

    return A, b, c


def test_unbounded():
    A = np.array([[-1, 1], [-2, -2], [-1, 4]])
    b = np.array([-1, -6, 2])
    c = np.array([1, 3])

    assert SIMPLEX_SOLVER.has_feasible_solution(A, b, c)
    assert (
        SIMPLEX_SOLVER.get_optimal_solution(A, b, c).res_code
        == ResultCode.UNBOUNDED_OPTIMAL
    )


def test_infeasible():
    A = np.array([[1], [-1]])
    b = np.array([5, -6])
    c = np.array([3])

    assert not SIMPLEX_SOLVER.has_feasible_solution(A, b, c)
    assert (
        SIMPLEX_SOLVER.get_optimal_solution(A, b, c).res_code == ResultCode.INFEASIBLE
    )


def test_finite_feasible():
    # optimal solution is 10.5
    A = np.array([[1, 1, 2], [2, 0, 3], [2, 1, 3]])
    b = np.array([4, 5, 7])
    c = np.array([3, 2, 4])

    assert SIMPLEX_SOLVER.has_feasible_solution(A, b, c)
    optimal_res = SIMPLEX_SOLVER.get_optimal_solution(A, b, c)
    assert optimal_res.res_code == ResultCode.FINITE_OPTIMAL
    assert math.isclose(optimal_res.optimal_score, 10.5)


def test_random_cases():
    for case in range(RANDOM_CASES):
        A, b, c = generate_random_case()

        res = SIMPLEX_SOLVER.get_optimal_solution(A, b, c)

        expected_res = linprog(A_ub=A, b_ub=b, c=-c, method="revised simplex")

        if expected_res.status in LINPROG_RCODE_TO_RCODE:
            expected_res_code = LINPROG_RCODE_TO_RCODE[expected_res.status]
            assert expected_res_code == res.res_code
            if expected_res_code == ResultCode.FINITE_OPTIMAL:
                assert math.isclose(expected_res.fun, -res.optimal_score)
