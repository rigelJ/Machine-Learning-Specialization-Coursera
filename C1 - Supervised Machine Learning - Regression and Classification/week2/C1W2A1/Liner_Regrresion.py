import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
import os 

import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

#change dir
#os.chdir('D:/Git/Machine-Learning-Specialization-Coursera/C1 - Supervised Machine Learning - Regression and Classification/week2/C1W2A1')

x_train, y_train = load_data()

#show the type of x_train and y_train
#print("Type of y_train:",type(x_train))
#print("First five elements of y_train are:\n", x_train[:33]) 

#visualize the data 
#plt.scatter(x_train,y_train,marker="x",c='r')
# Set the title
#plt.title("Profits vs. Population per city")
# Set the y-axis label
#plt.ylabel('Profit in $10,000')
# Set the x-axis label
#plt.xlabel('Population of City in 10,000s')
#plt.show()



# Not use the scikit learn
#----------------------------------------------------------
def Compute_cost(x,y,w,b):
    m=x.shape[0]

    cost = 0
    for i in range(m):
        f_wb = np.dot(x[i],w)+b
        cost = cost + (f_wb-y[i])**2
    cost = cost/(2*m)
    return cost

def Compute_gradient(x,y,w,b):
    m=x.shape[0]
    
    d_jw = 0
    d_jb = 0

    for i in range(m):
        f_wb = np.dot(x[i],w)+b
        d_jw = d_jw + (f_wb-y[i])*x[i]
        d_jb = d_jb + (f_wb-y[i])
    
    d_jw = d_jw/m
    d_jb = d_jb/m

    return d_jw,d_jb

def Gradient_descent(x,y,w_in,b_in,cost_function,gradient_function,alpha,num_iters):
    m = x.shape[0]

    J_history = []
    w_history = []
    w = copy.deepcopy(w_in) 
    b = b_in

    for j in range(num_iters):
        d_jw,d_jb = gradient_function(x,y,w,b)
        w = w - alpha*d_jw
        b = b - alpha*d_jbs
        if j<100000:
            cost = cost_function(x,y,w,b)
            J_history.append(cost)
        if j%math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {j:4}: Cost {float(J_history[-1]):8.2f} ")
    return w, b, J_history, w_history #return w and J,w history for graphing

"""
# Public tests
  
##Compute_cost test
from public_tests import *
#compute_cost_test(compute_cost)
#compute_gradient_test(Compute_gradient)



# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = Gradient_descent(x_train ,y_train, initial_w, initial_b, 
                     Compute_cost, Compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)


m=x_train.shape[0]
predicted=np.dot(x_train,w)+b 

plt.scatter(x_train,y_train,marker='o',c='r')
plt.plot(x_train,predicted,c='b')
# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()
"""
#----------------------------------------------------------------------------------------
#Use scikit-learn
#-----------LinearRegression------------------
linear_model = LinearRegression()
#X must be a 2-D Matrix
linear_model.fit(x_train.reshape(-1,1),y_train)
b = linear_model.intercept_
w = linear_model.coef_
#print(f"w={w:},b={b:0.2f}")
#print(f"'manual' prediction: f_wb = wx+b : {1200*w + b}")
y_pred = linear_model.predict(x_train.reshape(-1, 1))
#print("Prediction on training set:", y_pred)
X_test = np.array([[1200]])
#print(f"Prediction for 1200 sqft house: ${linear_model.predict(X_test)[0]:0.2f}")
#------------Norm-------------------------------s
scaler = StandardScaler()
x_norm = scaler.fit_transform(x_train.reshape(-1,1))
print(f"Peak to Peak range by column in Raw        X:{np.ptp(x_train,axis=0)}")  
print(f"Peak to Peak range by column in Normalized X:{np.ptp(x_norm,axis=0)}")

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(x_norm,y_train)
print(sgdr)

n_iter = sgdr.n_iter_
w_num = sgdr.t_
b_norm = sgdr.intercept_
w_norm = sgdr.coef_

print(f"model parameters:  w: {w_norm}, b:{b_norm}")

y_pred_sgd = sgdr.predict(x_norm)

y_pred = np.dot(x_norm, w_norm) + b_norm  

print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")
##plot predictions
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()