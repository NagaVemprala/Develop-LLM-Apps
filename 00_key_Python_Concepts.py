"""
00_key_Python_Concepts.py
=========================

A runnable primer for the core Python concepts used throughout the other
scripts in this repository. For every concept you will find:

    1. A short, high-level explanation (what the idea is and why it matters).
    2. One or more small, runnable examples right beneath the explanation.

Run this file directly from a terminal to see every example in action:

    python 00_key_Python_Concepts.py

Concepts covered
----------------
    1. Function parameters: defaults, keyword-only args, *args and **kwargs
    2. Unpacking and merging dictionaries with **
    3. Lambda (anonymous) functions
    4. Closures: functions that remember their enclosing scope
    5. Generators and the reusability of range()
    6. Decorators: the @ syntax, forwarding arguments, functools.wraps,
       and stacking decorators
"""

import time
from functools import wraps


# ============================================================
# 1. FUNCTION PARAMETERS: DEFAULTS, KEYWORD-ONLY ARGS, *args, **kwargs
# ============================================================
#
# Python functions can declare parameters in several ways, and the order in
# the signature matters:
#
#   def name(required, default=..., *args, keyword_only=..., **kwargs)
#
#   - required ......... must always be supplied (positionally or by keyword)
#   - default=value .... optional; falls back to value when not supplied
#   - *args ............ collects any EXTRA positional arguments into a tuple
#   - keyword_only ..... can only be passed by name (anything after *args)
#   - **kwargs ......... collects any EXTRA keyword arguments into a dict
#
# Two rules worth remembering:
#   * '**kwargs' must always be the LAST thing in a signature.
#   * Anything after '*args' is keyword-only (you cannot pass it
#     positionally).

def handle_request(url, timeout=30, *args, method="request_type",
                   mashup="mashup", **kwargs):
    """Demo signature mixing all parameter kinds.

    url and timeout are the standard arguments; extra positional values land
    in args; method/mashup are keyword-only with defaults; anything else
    passed by keyword lands in kwargs.
    """
    print(f"URL:      {url}")
    print(f"Timeout:  {timeout}")
    print(f"Method:   {method}")
    print(f"Mashup:   {mashup}")
    print(f"*args:    {args}")      # tuple of extra positional arguments
    print(f"**kwargs: {kwargs}")    # dict of extra keyword arguments


def demo_function_parameters():
    # Case 1: only what we need is passed; everything else uses its default
    # and the trailing "another_val=..." is captured by **kwargs.
    handle_request(url="https://example.com/get", method="GET",
                   another_val="Test")
    print("-" * 78)

    # Case 2: url is passed positionally, 20 binds to timeout, then 100 and
    # 200 are "extra" positional values so they land in *args; source=...
    # is an unknown keyword so it lands in **kwargs.
    handle_request("https://example.com/put", 20, 100, 200, method="PUT",
                   mashup="Custom mashup", source="ETL")

    # --- Unpacking a dictionary into a function call -----------------------
    # **d inside a call "splats" each dict key into a keyword argument.
    # This is very common when passing config payloads around. The function
    # only needs arg1..arg3; the extra key is absorbed by **kwargs.
    def print_kwargs(arg1, arg2, arg3, **kwargs):
        print("Named params:", arg1, arg2, arg3)
        print("Extra kwargs:", kwargs)

    payload = {"arg1": "value1", "arg2": "value2",
               "arg3": "value3", "extra_arg": "extra_value"}
    print("Calling print_kwargs(**payload):")
    print_kwargs(**payload)


# ============================================================
# 2. UNPACKING AND MERGING DICTIONARIES WITH **
# ============================================================
#
# The same ** operator works in dictionary literals: {**a, **b} builds a NEW
# dictionary containing the key/value pairs of a and then b. If both dicts
# share a key, the later dict (b) wins.
#
# Contrast that with dict.update(other), which merges IN PLACE: it mutates
# the original dictionary and returns None -- so don't assign its result!

def demo_dict_merging():
    dict1 = {"arg1": "value1", "arg2": "value2"}
    dict2 = {"arg2": "OVERWRITTEN", "arg4": "value4"}

    merged = {**dict1, **dict2}      # new dict; dict2's arg2 wins
    print("dict1   =", dict1)
    print("dict2   =", dict2)
    print("{**dict1, **dict2} =", merged)
    print("dict1 was NOT changed:", dict1)

    # .update() merges in place and returns None (a classic bug source).
    dict1.update(dict2)
    print("After dict1.update(dict2), dict1 =", dict1)


# ============================================================
# 3. LAMBDA (ANONYMOUS) FUNCTIONS
# ============================================================
#
# A lambda is a tiny, single-expression function without a name or def block.
# Reach for it when you need a throwaway function in one place (sorting keys,
# map/filter callbacks); if the logic needs more than one expression or a
# docstring, use a regular def instead.

def demo_lambdas():
    # A lambda is just a function object:
    double = lambda x: x * 2          # noqa: E731 (intentional demo)
    print("double(21) =", double(21))

    # Its most useful home: callbacks like sorted(key=...). Sort words by
    # length without defining a named function:
    words = ["generator", "lambda", "closure", "decorator", "dict"]
    by_length = sorted(words, key=lambda w: len(w))
    print("Words sorted by length:", by_length)

    # map() applies a function to every element of an iterable:
    squares = list(map(lambda x: x ** 2, [1, 2, 3, 4]))
    print("Squares of 1..4:", squares)


# ============================================================
# 4. CLOSURES: FUNCTIONS THAT REMEMBER THEIR ENCLOSING SCOPE
# ============================================================
#
# When a nested function references a variable from the function that defined
# it, it "closes over" that variable: the value stays alive (remembered) even
# after the outer function has returned. Closures are a neat way to create
# small stateful helpers or specialized functions on the fly.
#
#   - If the inner function only READS the outer variable, no extra keyword
#     is needed (see make_multiplier).
#   - If the inner function must REBIND it, declare 'nonlocal' so Python
#     knows the variable belongs to the enclosing scope (see make_counter).

def make_multiplier(factor):
    """Return a function that multiplies any input by factor."""
    def multiply(x):
        return x * factor      # factor is "remembered" here
    return multiply


def make_counter(start=0):
    """Return a stateful counter. Each call bumps the remembered count."""
    count = start
    def increment():
        nonlocal count         # allow rebinding count from the outer scope
        count += 1
        return count
    return increment


def demo_closures():
    double = make_multiplier(2)   # each closure remembers its own factor
    triple = make_multiplier(3)
    print("double(10) =", double(10))
    print("triple(10) =", triple(10))

    counter_a = make_counter()
    counter_b = make_counter(100)
    print("counter_a:", counter_a(), counter_a(), counter_a())  # 1, 2, 3
    print("counter_b (independent):", counter_b())              # 101


# ============================================================
# 5. GENERATORS AND THE REUSABILITY OF range()
# ============================================================
#
# A generator produces values LAZILY, one at a time, instead of building a
# whole list in memory. That makes generators memory-efficient for large
# streams -- but also single-use: a generator is "exhausted" once you have
# consumed its values, so iterating it a second time yields nothing.
#
# range() looks similar but is NOT a generator: it is a reusable iterable
# that can be turned into a list as many times as you like.
#
# Two ways to write a generator:
#   * generator expression:  (x for x in ...)
#   * generator function:    uses yield instead of return

def countdown(n):
    """Generator function: yields n, n-1, ..., 1 on demand."""
    while n > 0:
        yield n
        n -= 1


def demo_generators():
    # Generator expression -- note the parentheses, not brackets.
    gen = (x for x in range(3))
    print("First pass over generator:", list(gen))
    print("Second pass over generator:", list(gen))  # [] -- it is exhausted!

    # range() can be reused indefinitely.
    r = range(3)
    print("First pass over range:", list(r))
    print("Second pass over range:", list(r))        # still works!

    # A generator function behaves the same way and stays lazy.
    print("countdown(3) ->", list(countdown(3)))

    # Why care? Laziness keeps memory flat even for huge ranges:
    big = (x * x for x in range(10_000_000))
    print("Lazy generator, first three squares:",
          next(big), next(big), next(big))


# ============================================================
# 6. DECORATORS
# ============================================================
#
# A decorator is a function that takes another function and returns a
# modified version of it -- a reusable way to add behavior (logging, timing,
# access checks...) WITHOUT touching the wrapped function's code.
#
#   @my_decorator
#   def f(): ...
#
# is just clean sugar for:
#
#   def f(): ...
#   f = my_decorator(f)
#
# Three rules for writing correct decorators:
#   1. The inner wrapper must accept (*args, **kwargs) so it works for any
#      function signature.
#   2. The wrapper must return the result so the original return value is
#      not swallowed.
#   3. Decorate the wrapper with @wraps(func) (from functools) to copy the
#      original function's name/docstring onto the wrapper.

def log_calls(func):
    """Decorator that prints messages before and after the wrapped call."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("  -> something happens BEFORE the function runs")
        result = func(*args, **kwargs)
        print("  -> something happens AFTER the function runs")
        return result
    return wrapper


def timer(func):
    """Decorator that measures and prints the wrapped function's runtime."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  -> '{func.__name__}' took {elapsed:.6f} seconds")
        return result
    return wrapper


def decorator_one(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("  Decorator One: before")
        result = func(*args, **kwargs)
        print("  Decorator One: after")
        return result
    return wrapper


def decorator_two(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("  Decorator Two: before")
        result = func(*args, **kwargs)
        print("  Decorator Two: after")
        return result
    return wrapper


def demo_basic_decorator():
    @log_calls
    def say_hello(name="World"):
        print(f"     Hello, {name}!")

    print("Calling @log_calls say_hello('Alex'):")
    say_hello("Alex")

    # Without the @ shorthand you would write the exact same thing by hand:
    #   say_hello = log_calls(say_hello)


def demo_timer_decorator():
    @timer
    def heavy_calculation(n):
        """Sum of squares from 0 up to n."""
        return sum(i * i for i in range(n))

    print("Calling @timer heavy_calculation(1_000_000):")
    total = heavy_calculation(1_000_000)
    print("  -> result:", total)


def demo_wraps():
    # A decorator written WITHOUT @wraps clobbers the original metadata.
    def bare_decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @bare_decorator
    def named_function():
        """Returns the number 42."""
        return 42

    print("Without @wraps -> name:", repr(named_function.__name__),
          "| doc:", repr(named_function.__doc__))

    # log_calls uses @wraps, so metadata survives.
    @log_calls
    def named_function_too():
        """Returns the number 42."""
        return 42

    print("With @wraps    -> name:", repr(named_function_too.__name__),
          "| doc:", repr(named_function_too.__doc__))


def demo_stacked_decorators():
    # Decorators are applied BOTTOM-UP (closest to the function first):
    #   my_func = decorator_two(decorator_one(my_func))
    # so decorator_one runs nearest to the original body.
    @decorator_two
    @decorator_one
    def announce():
        print("     [original function body runs]")

    print("Calling a function decorated with @decorator_two @decorator_one:")
    announce()


# ============================================================
# MAIN
# ============================================================

def section(title):
    """Print a clearly visible banner between tutorial sections."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    section("1. Function parameters: defaults, keyword-only args, "
            "*args and **kwargs")
    demo_function_parameters()

    section("2. Unpacking and merging dictionaries with **")
    demo_dict_merging()

    section("3. Lambda (anonymous) functions")
    demo_lambdas()

    section("4. Closures: functions that remember their enclosing scope")
    demo_closures()

    section("5. Generators and the reusability of range()")
    demo_generators()

    section("6. Decorators")
    demo_basic_decorator()
    demo_timer_decorator()
    demo_wraps()
    demo_stacked_decorators()

    print("\n" + "=" * 78)
    print("All sections completed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
