import time
from dataclasses import dataclass
import numpy as np
from test_simplex import generate_random_case
from Simplex import Simplex
from scipy.optimize import linprog

CASES_AMOUNT = 10000
MAX_M = 15
MAX_N = 10
MIN_INT_ELEMENT = -2147483648
MAX_INT_ELEMENT = 2147483647

SIMPLEX_SOLVER = Simplex()

FUNCS_TO_ANALYZE = {
    "RevisedSimplex's full revised simplex": SIMPLEX_SOLVER.get_optimal_solution,
    "RevisedSimplex's feasible solution check": SIMPLEX_SOLVER.has_feasible_solution,
    "Scipy's revised simplex method": lambda a, b, c: linprog(
        A_ub=a, b_ub=b, c=-c, method="revised simplex"
    ),
}


@dataclass
class TestCase(object):
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray


def get_random_cases(cases_amount: int = CASES_AMOUNT):
    cases = []
    for case in range(cases_amount):
        A, b, c = generate_random_case(MAX_M, MAX_M, MIN_INT_ELEMENT, MAX_INT_ELEMENT)
        cases.append(TestCase(A, b, c))
    return cases


def measure_time(test_cases, func):
    start_time = time.time()
    for case in test_cases:
        func(case.A, case.b, case.c)
    return time.time() - start_time


def compare_performance():
    cases = get_random_cases()
    performances = {}
    for func_desc, func in FUNCS_TO_ANALYZE.items():
        performances[func_desc] = measure_time(cases, func)

    for func_desc, duration in performances.items():
        print(
            f"The total duration of running {func_desc} on {CASES_AMOUNT}"
            f" random cases: {duration}s"
        )


if __name__ == "__main__":
    compare_performance()
