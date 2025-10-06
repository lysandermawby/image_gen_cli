#!/usr/bin/python
"""utility functions"""

def decimal_to_base(val, new_base):
    """convert from a decimal representation to an octal representation"""

    val = int(val)
    if val == 0:
        return 0 
    
    output_val = ""
    while val > 0:
        output_val = str(val % new_base) + output_val
        val //= new_base
    
    # return a string octal value
    return output_val
