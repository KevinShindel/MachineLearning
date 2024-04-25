"""
Description : This module provide stress loading testing for the base_using module. For predict most faster function
Author : Kevin Shindel
Date : 2024-24-04
"""

from Pandas.base_using import dict_inclusion, dict_inclusion_setdefault, dict_inclusion_defaultdict
from unittest import TestCase, main


def time_wrapper():
    def wrapper(func):
        def inner(*args, **kwargs):
            import time
            start = time.time()
            func(*args, **kwargs)
            print(f'{func.__name__} took {time.time() - start} seconds')
        return inner
    return wrapper


class PandasTest(TestCase):

    def setUp(self):
        self.mock_data = list(self.mock_data())

    @staticmethod
    def mock_data(num_fruits: int = 100000):
        # generate fruits data
        for _ in range(num_fruits):
            yield 'apple'
            yield 'banana'
            yield 'cherry'
            yield 'pineapple'
            yield 'mango'

    @time_wrapper()
    def test_dict_inclusion(self):
        """ best time 0.064 seconds """
        dict_inclusion(self.mock_data)


    @time_wrapper()
    def test_dict_inclusion_defaultdict(self):
        """ best time 0.049 seconds """
        dict_inclusion_defaultdict(self.mock_data)

    @time_wrapper()
    def test_dict_inclusion_setdefault(self):
        """ best time 0.076 seconds """
        dict_inclusion_setdefault(self.mock_data)


if __name__ == '__main__':
    main()
