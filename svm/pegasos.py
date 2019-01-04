import util
import pickle
from collections import Counter
import time
import matplotlib.pyplot as plt


def pegasos(X_train, y_train, lambda_reg=1, max_it=1000, tol=1e-6):
    w = Counter()
    t = 1
    epoch = 1
    objective = 1e5
    objective2 = 10
    m = len(y_train)
    
    while abs(objective-objective2)>tol and epoch <= max_it:
        objective2 = objective
        objective = 0
        for j in range(m):
            t = t + 1
            step = 1/(t*lambda_reg)
            review = X_train[j]
            result = y_train[j]
            scale = -(step*lambda_reg)
            cond = result*util.dotProduct(w, review)
            
            if cond < 1:
                util.increment(w, scale, w)
                util.increment(w, step*result, review)
            else:
                util.increment(w, scale, w)

            objective += max(0,1-cond)
            
        objective = objective/m
        objective = objective + lambda_reg/2 * util.dotProduct(w,w)
        epoch += 1

    return w


def pegasos_sw(X_train, y_train, lambda_reg=1, max_it=1000, tol=1e-4):
    W = Counter()
    s = 1
    t = 1
    epoch = 1
    objective = 1e5
    objective2 = 10
    m = len(y_train)
    
    while abs(objective-objective2)>tol and epoch <= max_it:
        objective2 = objective
        objective = 0
        for j in range(m):
            t = t + 1
            step = 1/(t*lambda_reg)
            review = X_train[j]
            result = y_train[j]
            scale = -(step*lambda_reg)
            cond = result*s*util.dotProduct(W, review)
            
            if cond < 1:
                s = (1+scale)*s
                util.increment(W, step*result/s, review)
            else:
                s = (1+scale)*s

            objective += max(0,1-cond)
            
        objective = objective/m
        objective = objective + lambda_reg/2 * (s**2) * util.dotProduct(W,W)
        epoch += 1

    return s,W


def dotProduct1(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct1(d2, d1)
    else:
        imp = []
        imp2 = []
        ret = 0
        for f, v in d2.items():
            a = d1.get(f, 0)
            tmp = a * v
            imp.append((tmp,a,v))
            imp2.append(tmp)
            ret += tmp
        return ret, imp, imp2


def percent_error(X_test, y_test, w):
    m = len(y_test)
    err = 0
    sign = lambda x: x and (1, -1)[x < 0]
    for i in range(m):
        review = X_test[i]
        result = y_test[i]
        prediction, wixi, wixi2 = dotProduct1(w, review)
        if sign(prediction) != result:
            print('prediction: ', prediction)
            print('result: ', result)
            print('review: ', review)
            print('wixi2: ', sorted(wixi2, key=abs))
            print('wixi: ', wixi)
            err += 1

    return err*100/m



def main():
    
    # import 2D list with schuffled reviews 
    with open("data_as_list", "rb") as f:
        review = pickle.load(f)

    # divide into training, test
    X_train, X_test, y_train, y_test = util.divide_train_test(review)

    # convert X into a list of sparse rep
    for j in range(1500):
       X_train[j] = util.convert_to_BOW(X_train[j])
       
    for i in range(500):
        X_test[i] = util.convert_to_BOW(X_test[i])


##    # run pegasos for timings
##    a = time.time()
##    w = pegasos(X_train,y_train,max_it=5)
##    b = time.time()
##    print('pegasos: ', b-a)
##    
##    c = time.time()
##    s,W = pegasos_sw(X_train,y_train,max_it=5)
##    for f, v in W.items():
##        W[f] = v * s
##    d = time.time()
##    print('pegasos_sw: ', d-c)
##
##    # check if both PEGs produce same result
##    for f,v in w.items():
##        if abs(w[f]-W.get(f,0))>1e-1:
##            print(f,v)
##    
##    # choose best hyperparameter
##    lambda_range =[1e-4,0.0005,1e-3,0.005,1e-2]
##    for l in lambda_range:
##        s,W = pegasos_sw(X_train,y_train,lambda_reg=l)
##        for f, v in W.items():
##            W[f] = v * s
##        p = percent_error(X_test, y_test, W)
##        print(l, p)
##        
##    # plot hyperparameter vs percent_error
##    lambdas = [ 1e-5, 1e-4, 0.0005, 1e-3, 0.005, 1e-2, 1e-1, 1]
##    err = [ 17.4, 16.8, 17.2, 16.4, 17.6, 18, 20.2, 20.8]
##    
##    plt.semilogx(lambdas, err)
##    plt.xlabel('$\lambda$')
##    plt.ylabel('percent error')
##    plt.title('Searching for hyperparameter $\lambda$ that minimizes percent error')
##    plt.grid(True)
##    plt.show()
    s,W = pegasos_sw(X_train,y_train,lambda_reg=0.001)
    for f, v in W.items():
        W[f] = v * s
    p = percent_error(X_test, y_test, W)
    print('err: ', p)
    print('w: ', W)

    
main()
