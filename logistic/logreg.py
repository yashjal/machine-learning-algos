import numpy as np
from sklearn import preprocessing
from scipy.optimize import minimize
import pickle
import matplotlib.pyplot as plt

def f_objective(theta, X, y, l2_param=1):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''
    xw = np.dot(X,np.reshape(theta,(-1,1)))
    J = 0
    n = y.size
    for i in range(n):
        J += np.logaddexp(0,-y[i]*xw[i,0])
    J /= n
    J += l2_param*np.dot(theta,theta)
    
    return J
    
    
def fit_logistic_reg(X_train, y, objective_function, l2_param=1):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter
        
    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''
    
    w0 = np.zeros(X_train.shape[1])
    res = minimize(objective_function, w0, args=(X_train, y, l2_param))
    
    return res.x

def main():
    READ = False
    LOAD = True

    if READ:
        X_train = np.loadtxt("X_train.txt", delimiter=',')
        X_val = np.loadtxt("X_val.txt", delimiter=',')
        y_train = np.loadtxt("y_train.txt", delimiter=',')
        y_val = np.loadtxt("y_val.txt", delimiter=',')
         # standardize X
        scaler = preprocessing.StandardScaler().fit(X_train)
        scaler.transform(X_train)
        scaler.transform(X_val)
        # Add bias term
        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1))))
        # map 0 y-values to -1
        reducer = lambda x: -1 if x == 0 else 1
        vfunc = np.vectorize(reducer)
        y_train = vfunc(y_train)
        y_val = vfunc(y_val)

        with open("x_tr", "wb") as f:
            pickle.dump(X_train, f)
        with open("x_vl", "wb") as g:
            pickle.dump(X_val, g)
        with open("y_tr", "wb") as h:
            pickle.dump(y_train, h)
        with open("y_vl", "wb") as i:
            pickle.dump(y_val, i)

    if LOAD:
        with open("x_tr", "rb") as f:
            X_train = pickle.load(f)
        with open("x_vl", "rb") as g:
            X_val = pickle.load(g)
        with open("y_tr", "rb") as h:
            y_train = pickle.load(h)
        with open("y_vl", "rb") as i:
            y_val = pickle.load(i)

    # finding best l2_param
    '''
    n = y_val.size
    lam = [0,1e-16,2e-16,3e-16,4e-16,5e-16,6e-16,7e-16,8e-16,9e-16,1e-15,2e-15,3e-15,4e-15,5e-15,6e-15,7e-15,8e-15,9e-15,1e-14,2e-14,3e-14,4e-14,5e-14,6e-14,7e-14,8e-14,9e-14,1e-13,2e-13,3e-13,4e-13,5e-13,6e-13,7e-13,8e-13,9e-13,1e-12,2e-12,3e-12,4e-12,5e-12,6e-12,7e-12,8e-12,9e-12,1e-11,2e-11,3e-11,4e-11,5e-11,6e-11,7e-11,8e-11,9e-11,1e-10,2e-10,3e-10,4e-10,5e-10,6e-10,7e-10,8e-10,9e-10,1e-9]
    log_likelihood = []
    for l in lam:
        w = fit_logistic_reg(X_train, y_train, f_objective, l)
        tmp = -n*f_objective(w,X_val,y_val,l)
        log_likelihood.append(tmp)
        print('lambda, log_likelihood: ', l, tmp)

    plt.semilogx(lam, log_likelihood)
    plt.title('Log-likelihood on validation set')
    plt.xlabel('l2-parameter')
    plt.ylabel('log-likelihood')
    plt.show()
    '''
    
    #optimal w
    w = fit_logistic_reg(X_train, y_train, f_objective, 3e-15)
    xw = -np.ravel(np.dot(X_val,np.reshape(w,(-1,1))))
    preds = 1/(1+np.exp(xw))
    print('preds: ', preds)
    
    # calibration plot
    cals = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    pos = np.zeros(11)
    ys = np.zeros(11)
    i = 0
    for p in preds:
        y = 1
        if y_val[i] == -1:
            y = 0
        f = round(p,1)
        if f == 1.0:
            pos[10] += 1
            ys[10] += y
        elif f == 0.9:
            pos[9] +=1
            ys[9] += y
        elif f == 0.8:
            pos[8] +=1
            ys[8] += y
        elif f == 0.7:
            pos[7] +=1
            ys[7] += y
        elif f == 0.6:
            pos[6] +=1
            ys[6] += y
        elif f == 0.5:
            pos[5] +=1
            ys[5] += y
        elif f == 0.4:
            pos[4] +=1
            ys[4] += y
        elif f == 0.3:
            pos[3] +=1
            ys[3] += y
        elif f == 0.2:
            pos[2] +=1
            ys[2] += y
        elif f == 0.1:
            pos[1] +=1
            ys[1] += y
        elif f == 0.0:
            pos[0] +=1
            ys[0] += y
        i += 1

    for k in range(len(ys)):
        if pos[k] != 0:
            ys[k] = ys[k]/pos[k]
    
    line_up, = plt.plot(cals,ys,'.-',label='Logistic')
    line_down, = plt.plot([0,1],[0,1],'k:',label='Perfectly calibrated')
    plt.legend(handles=[line_up, line_down])
    plt.title('Calibration plot on validation set')
    plt.xlabel('Approximate f(x)')
    plt.ylabel('Fraction of positives')
    plt.show()
    



main()
