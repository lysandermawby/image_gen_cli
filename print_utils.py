#!/usr/bin/python
"""utils for better printing"""

# colour codes
BLACK = "\033[0;30m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
MAGENTA = "\033[0;35m"
CYAN = "\033[0;36m"
WHITE = "\033[0;37m"

# reset colour attributes
NC = "\033[0m"

# print with a colour applied
def print_colour(text, colour):
    print(f"{colour}{text}{NC}")

def print_error(text):
    print_colour(text, RED)

def print_warning(text):
    print_colour(text, YELLOW)

def print_success(text):
    print_colour(text, GREEN)

def print_debug(text):
    print_colour(text, BLUE)

def print_info(text):
    print(text, MAGENTA)
