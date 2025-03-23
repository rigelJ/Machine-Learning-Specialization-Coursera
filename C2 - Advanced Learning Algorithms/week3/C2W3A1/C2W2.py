import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from public_tests_a1 import *

tf.keras.backend.set_floatx('float64')
from assigment_utils import *

tf.autograph.set_verbosity(0)

#Evaluating a Learning Algorithm(Ploynomial Regression)

##Generate some data
X,y,x_ideal,y_ideal = gen_data(18, 2, 0.7)
print("X_shape",X.shape,"y_shape",y.shape)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=1)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

##Plot Train,Test sets 
fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x_ideal,y_ideal,"-",color = "orangered",label = "y_ideal",lw=1)
ax.set_title("Plot Training Test",fontsize = 14)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.scatter(x_train,y_train,color="red",label="y_train")
ax.scatter(x_test,y_test,color = "blue",label = "y_test")
ax.legend(loc="upper left")
#plt.show()

##Build a class to modify the lin_model
class line_model:
    def __init__(self,degree,regularization=False,lamda=0):
        if regularization:
            self.linear_model = Ridge(alpha = lamda)
        else:
            self.linear_model = LinearRegression()
        self.poly = PolynomialFeatures(degree,include_bias=False)
        self.scaler = StandardScaler()
    def fit(self,x_train,y_train):
        x_train_mapped = self.poly.fit_transform(x_train.reshape(-1,1))
        x_train_mapped_scaled = self.scaler.fit_transform(x_train_mapped)
        self.linear_model.fit(x_train_mapped_scaled,y_train)

    def predict(self,x):
        x_mapped = self.poly.fit_transform(x.reshape(-1,1))
        x_mapped_scaled = self.scaler.transform(x_mapped)
        y_hat = self.linear_model.predict(x_mapped_scaled)
        return (y_hat)
    
    def mse(self,y,y_hat):
        err = mean_squared_error(y,y_hat)**2
        return (err)
 

##Compare performance on training and test data
"""
degree = 10
lmodel = lin_model(degree)
lmodel.fit(x_train, y_train)

yhat = lmodel.predict(x_train)
err_train = lmodel.mse(y_train, yhat)

yhat = lmodel.predict(x_test)
err_test = lmodel.mse(y_test, yhat)

print(f"training err {err_train:0.2f}, test err {err_test:0.2f}")

x = np.linspace(0,int(X.max()),100)  # predict values for plot
y_pred = lmodel.predict(x).reshape(-1,1)

plt_train_test(x_train, y_train, x_test, y_test, x, y_pred, x_ideal, y_ideal, degree)
###Build a high degree polynomial model to min train error
"""

#Bias and vaience

##Generate the data and split to train/cv/test

X,y, x_ideal,y_ideal = gen_data(40, 5, 0.7)
x_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.40, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.50, random_state=1)
print("X_train.shape", x_train.shape,"y_train.shape", y_train.shape)
print("X_cv.shape", x_cv.shape, "y_cv.shape", y_cv.shape)
print("X_test.shape", x_test.shape, "y_test.", y_test.shape)


##Plot train cv test
fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
ax.set_title("Training, CV, Test",fontsize = 14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(x_train, y_train, color = "red",           label="train")
ax.scatter(x_cv, y_cv,       color = dlc["dlorange"], label="cv")
ax.scatter(x_test, y_test,   color = dlc["dlblue"],   label="test")
ax.legend(loc='upper left')
#plt.show()

##Iterrate to find the optimal degree 

max_degree = 9
err_train = np.zeros(max_degree)
err_cv = np.zeros(max_degree)
x = np.linspace(0,int(X.max()),100)
y_pred = np.zeros((100,max_degree)) 

for degree in range(max_degree):
    lmodel = line_model(degree+1)
    lmodel.fit(x_train,y_train)
    y_hat = lmodel.predict(x_train)
    err_train[degree] = (lmodel.mse(y_train,y_hat))
    y_hat = lmodel.predict(x_cv)
    err_cv[degree] = (lmodel.mse(y_cv,y_hat))
    y_pred[:,degree] = lmodel.predict(x)

optimal_degree = np.argmin(err_cv)+1

plt.close("all")
#plt_optimal_degree(x_train, y_train, x_cv, y_cv, x, y_pred, x_ideal, y_ideal, 
                  # err_train, err_cv, optimal_degree, max_degree)

## Iterate to find the optinmal Regularization lambda

lambda_range = np.array([0.0, 1e-6, 1e-5, 1e-4,1e-3,1e-2, 1e-1,1,10,100])
num_steps = len(lambda_range)
degree = 10
err_trains = np.zeros(num_steps)
err_cvs = np.zeros(num_steps)
x = np.linspace(0,int(X.max()),100)
y_pred = np.zeros((100,num_steps))

print(y_pred.shape)

for i in range(num_steps):
    lamda_now = lambda_range[i]
    lmodel = line_model(degree,regularization=True,lamda=lamda_now)
    lmodel.fit(x_train,y_train)
    y_hat = lmodel.predict(x_train)
    err_trains[i] = lmodel.mse(y_train,y_hat)
    y_hat = lmodel.predict(x_cv)
    err_cvs[i] = lmodel.mse(y_cv,y_hat)
    y_pred[:,i] = lmodel.predict(x)

optimal_reg_idx = np.argmin(err_cvs)

print(f"the optinal reg lambda is {lambda_range[optimal_reg_idx]}")

#plt.close("all")
#plt_tune_regularization(x_train, y_train,x_cv, y_cv, x, y_pred, err_train, err_cv, optimal_reg_idx, lambda_range)

##Increasing the data set
x_train, y_train, x_cv, y_cv, x, y_pred, err_train, err_cv, m_range,degree = tune_m()
#plt_tune_m(x_train, y_train, x_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree)

#Ealuate a learning algoritm in Neural netwok

##Generate data
X, y, centers, classes, std = gen_blobs() 
x_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.50, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.20, random_state=1)
print("X_train.shape:", x_train.shape, "X_cv.shape:", x_cv.shape, "X_test.shape:", x_test.shape)

#plt_train_eq_dist(x_train, y_train,classes, X_cv, y_cv, centers, std)

##Some evaluting function to classify the error
def eval_cat_err(y,y_hat):
    inconum = 0
    for i in range(len(y)):
        if y[i] != y_hat[i]:
            inconum+=1
        else:
            continue
    return inconum/len(y)

#More complex model evalute

##set up the model 
import logging 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

tf.random.set_seed(1234)

##Complex model
"""
model = Sequential(
    [
        tf.keras.layers.Dense(120,activation="relu"),
        tf.keras.layers.Dense(40,activation="relu"),
        tf.keras.layers.Dense(6,activation="linear")
    ], name="Complex"
)

model.compile(
    loss = SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)

model.fit(
    x_train,y_train,
    epochs=1000
)

model.summary()

y_hat = np.argmax(tf.nn.softmax(model.predict(x_train)).numpy(),axis=1)
print(y_hat[:6])

y_hat2 = np.argmax(tf.nn.softmax(model.predict(x_cv)).numpy(),axis=1)
print(y_hat2[:6])

train_err = eval_cat_err(y_train,y_hat)
cv_err = eval_cat_err(y_cv,y_hat2)

print(f"The train_err is {train_err}")
print(f"The cv_err is {cv_err}")"
"""

#SHOW THAT the err_cv is 0.1 bigger than err_train 0.0075  so the model is overfit

##Simple model
"""
tf.random.set_seed(1234)
model_s = Sequential(
    [
        tf.keras.layers.Dense(6,activation="relu",name="L1"),
        tf.keras.layers.Dense(6,activation="linear",name="L2")
    ],name="Simple"
)

model_s.compile(
    loss = SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
)

model_s.fit(
    x_train,y_train,
    epochs = 1000
)

model_s.summary()

y_hat = np.argmax(tf.nn.softmax(model_s.predict(x_train)).numpy(),axis=1)
print(y_hat[:6])

y_hat2 = np.argmax(tf.nn.softmax(model_s.predict(x_cv)).numpy(),axis=1)
print(y_hat2[:6])

train_err = eval_cat_err(y_train,y_hat)
cv_err = eval_cat_err(y_cv,y_hat2)

print(f"The train_err is {train_err}")
print(f"The cv_err is {cv_err}")

#Train err is 0.07 cv err is also 0.07 "
""
"""

#Regularization use to moderate the complex model

"""
tf.random.set_seed(1234)
model_r = Sequential(
    [
        ### START CODE HERE ### 
        tf.keras.layers.Dense(120, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        tf.keras.layers.Dense(40, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        tf.keras.layers.Dense(6, activation="linear")
        ### START CODE HERE ### 
    ], name= "ComplexRegularized"
)
model_r.compile(
    ### START CODE HERE ### 
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    ### START CODE HERE ###
)

model_r.fit(
    x_train,y_train,
    epochs=1000
)

model_r.summary()

y_hat = np.argmin(tf.nn.softmax(model_r.predict(x_train)).numpy(),axis=1)
y_hat2 = np.argmin(tf.nn.softmax(model_r.predict(x_cv)).numpy(),axis=1)
train_err = eval_cat_err(y_train,y_hat)
cv_err = eval_cat_err(y_cv,y_hat2)

print(f"The train_err is {train_err}")
print(f"The cv_err is {cv_err}")"
"
"""

##Iterate to find the optimal regularization value

tf.random.set_seed(1234)
lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3] 
models = [None]*len(lambdas)

for i in range(len(lambdas)):
    lambda_now = lambdas[i]
    models[i] = Sequential(
        [
            tf.keras.layers.Dense(120,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_now)),
            tf.keras.layers.Dense(40,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_now)),
            tf.keras.layers.Dense(classes,activation="linear")
        ],name = "iterReg"
    ) 

    models[i].compile(
        loss = SparseCategoricalCrossentropy(from_logits=True),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    )

    models[i].fit(
        x_train,y_train,
        epochs=1000
    )
    
print("Finish compiled!")


