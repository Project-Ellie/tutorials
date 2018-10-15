import sys
import numpy as np
# A little tool stolen from Magnus Erik Hvass Pedersen
def print_progress(format_string, count, total=1.0):

    pct_complete = float(count) / total
    format_string = "\r"+ format_string
    msg = format_string.format(pct_complete)

    sys.stdout.write(msg)
    sys.stdout.flush()
    
    
# returns an np.array the size of 'array' with booleans True if that value in array is included in 'values'
def array_in(array, values):
    return np.array([ 
        np.array([ j[0] == i[0] and j[1] == i[1] 
                  for i in values]).any() 
        for j in array ])