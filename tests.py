from forecasting import algorithms
import random

def test_sliding_window_one():
    x, y = algorithms._sliding_window(list(range(100)), 5)
    assert x[0] != x[-1]

def test_sliding_window_two():
    x, y = algorithms._sliding_window(list(range(100)), 5)
    first_index = random.randint(0,99)
    second_index = random.randint(0,99)
    while first_index == second_index:
        second_index = random.randint(0,99)
    assert x[first_index] != x[second_index]

def test_sliding_window_three():
    x,y = algorithms._sliding_window(list(range(100)), 5)
    assert len(x) == len(y)
