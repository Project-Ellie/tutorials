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

def one_hot(df, size):
    ords = list(df)
    return np.transpose(np.eye(size)[ords])

def one_hot(df, size):
    ords = list(df)
    return np.transpose(np.eye(size)[ords])


def create_input_data(df, select_feats=[], oh_feats={}, cross_feats=[]):    
    """
    create a list of input columns from pandas raw data
    df: a pandas dataframe containing raw input data
    select_feats: an array containing the names of features to be selected without transformation
    oh_feats: a dictionary containing the names and sizes of discrete numerical features that are to be one-hot encoded
    cross_feats: a list of oh_feats consisting of two discrete features to cross
    """

    def _safe_append(l, r):
        if l == [] or l is None:
            return r
        else:
            return np.append(l, r, axis=0)

    res = [list(df[n]) for n in select_feats]
    
    for k in oh_feats:
        res = _safe_append(res, one_hot(df[k], oh_feats[k]))

    for c in cross_feats:
        lk, ls = c.items()[0]
        rk, rs = c.items()[1]
        lhs = one_hot(df[lk], ls)
        rhs = one_hot(df[rk], rs)
        cross = [(lhs[:,i].reshape(ls,1) * rhs[:,i].reshape(1,rs)).reshape(rs*ls) for i in range(len(df))]
        cross = np.transpose(cross)
        res = _safe_append(res, cross)

    return res


    
