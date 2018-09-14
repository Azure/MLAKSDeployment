# Copyright (C) Microsoft Corporation.  All rights reserved.

from __future__ import print_function
import timeit


def elapsed(func, return_elapsed=False):
    """A decorator that times function execution."""
    if __debug__:
        def wrapper(*args, **kwargs):
            beg_ts = timeit.default_timer()
            retval = func(*args, **kwargs)
            end_ts = timeit.default_timer()
            elapsed_time = end_ts - beg_ts
            if return_elapsed is True:
                return elapsed_time
            print('%s elapsed: %f' % (func.__name__, end_ts - beg_ts))
            return retval
        return wrapper
    else:
        return func
