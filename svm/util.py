# Taken from http://web.stanford.edu/class/cs221/ Assignment #2 Support Code
from collections import Counter

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale


def convert_to_BOW(review):
    '''
    review: 1D review list with each element being a single word from a review
    '''
    cnt = Counter()
    for word in review:
        cnt[word] += 1

    return cnt
    
def divide_train_test(review):
    '''
    review: 2D list with each row containing words from a review and score
    '''
    #convert review list into desired data list
    #initialize X, y
    X_train = review[:1500]
    X_test = review[1500:]
    y_train = [0] *1500
    y_test = [0] *500
    i=0
    for rev in review:
        if i < 1500:
            y_train[i] = X_train[i][-1]
            X_train[i] = X_train[i][:-1]
        else:
            y_test[i-1500] = X_test[i-1500][-1]
            X_test[i-1500] = X_test[i-1500][:-1]
            
        i = i+1

    return X_train, X_test, y_train, y_test

