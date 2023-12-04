from typing import Tuple
import functools 
import warnings
import inspect


def deprecated(*, ymd: Tuple[int] = None, optional_message: str = None):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn("Function {} is going to be deprecated".format(func.__name__) + " after %d-%d-%d." % ymd if ymd else '.',
                        category=DeprecationWarning,
                        stacklevel = 2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return func(*args, **kwargs)
        return wrapper 
    return decorate
        
def typed(types: dict = None, *, optional_message: str = None):
    def decorate(func):
        arg_names = inspect.getfullargspec(func)[0]
        @functools.wraps(func)
        def wrapper(*args,**kwargs):
            if types:
                for name, arg in zip(arg_names, args):
                    if name in types and not isinstance(arg, types[name]):
                        raise TypeError("Argument {} must be {}".format(name, types[name]))
                for name, arg in kwargs.items():
                    if name in types and not isinstance(arg, types[name]):
                        raise TypeError("Argument {} must be {}".format(name, types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate
                
                
def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Function {} took {} seconds to execute".format(func.__name__, end - start))
        return result
    return wrapper

def memoize(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args not in wrapper.cache:
            wrapper.cache[args] = func(*args, **kwargs)
        return wrapper.cache[args]
    wrapper.cache = {}
    return wrapper

def count_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.num_calls += 1
        print("Call {} of {}".format(wrapper.num_calls, func.__name__))
        return func(*args, **kwargs)
    wrapper.num_calls = 0
    return wrapper
