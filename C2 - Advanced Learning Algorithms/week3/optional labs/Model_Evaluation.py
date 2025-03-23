import numpy as np

#for building linear regression models and preparing data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#for plot data
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#for building and training neural network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)


# custom function
import utils

np.set_printoptions(precision=2)

#change os dir

"""
##--------Regression---------------

#load data
data= np.loadtxt('./data/data_w3_ex1.csv',delimiter=',')

x= data[:,0]
y= data[:,1]

x=np.expand_dims(x,axis=1)
y=np.expand_dims(y,axis=1)


#plot the entire dataset
plt.rcParams["figure.figsize"]=(12,8)
plt.rcParams["lines.markersize"]=12

plt.scatter(x,y,marker='x',c='r')
plt.title("input vs. target")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


#Get 60% trainingset 20% cvset 20% testset
x_train,x_,y_train,y_ = train_test_split(x,y,test_size=0.4,random_state=1)
x_cv,x_test,y_cv,y_test = train_test_split(x_,y_,test_size=0.5,random_state=1)

del x_,y_


#plot the trainset cvset testset
plt.rcParams['figure.figsize']=(10,10)
plt.rcParams["lines.markersize"]=8
plt.scatter(x_train,y_train,marker='x',c='r',label="training")
plt.scatter(x_cv,y_cv,marker='o',c='b',label='cross validation')
plt.scatter(x_test,y_test,marker='^',c='g',label='test')
plt.title('train cv test')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#feature scaling use StandardScalder
scaler_linear = StandardScaler()

x_train_scaled = scaler_linear.fit_transform(x_train)

#utils.plot_dataset(x=train_scaled, y=y_train, title="scaled input vs. target")

#train the model
linear_model = LinearRegression()

linear_model.fit(x_train_scaled,y_train)

#evaluate the model
y_hat = linear_model.predict(x_train_scaled)

train_MSE = mean_squared_error(y_train,y_hat)/2
print(f"Training MSE(use sklearn function) is {train_MSE}")

total_square_error = 0
for i in range(len(y_hat)):
    square_error_i = (y_hat[i] - y_train[i])**2
    total_square_error += square_error_i
train_mse = total_square_error/(2*len(y_hat))
#print(f"Training MSE(use def function)is {train_mse.squeeze()}")

# Caculate the mse in cv set,scaling and get the MSE 
x_cv_scaled = scaler_linear.transform(x_cv)

y_hat = linear_model.predict(x_cv_scaled)

cv_MSE = mean_squared_error(y_cv,y_hat)/2
print(f"CV MSE(use sklearn function) is {cv_MSE}")

total_square_error = 0
for i in range(len(y_hat)):
    square_error_i = (y_hat[i]-y_cv[i])**2
    total_square_error += square_error_i
cv_mse =total_square_error/(2*len(y_hat))


#print(f"cv mse(use def funtion) is {cv_mse.squeeze()}")


#Create the additional features
ploy = PolynomialFeatures(degree=2,include_bias=False)
x_train_mapped = ploy.fit_transform(x_train)

scaler_poly = StandardScaler()
x_train_mapped_scaled = scaler_poly.fit_transform(x_train_mapped)

#Training model
model = LinearRegression()

model.fit(x_train_mapped_scaled,y_train)

y_hat = model.predict(x_train_mapped_scaled)

train_MSE = mean_squared_error(y_train,y_hat)/2

print(f"Training MSE for 2 ploy is {train_MSE}")

#Get the CV MSE in 2ploy ploy and scaling and MSE

x_cv_mapped = ploy.transform(x_cv)

x_cv_mapped_scaled = scaler_poly.transform(x_cv_mapped)

y_hat = model.predict(x_cv_mapped_scaled)

cv_MSE = mean_squared_error(y_cv,y_hat)/2

print(f"cv MSE for 2 ploy is {cv_MSE}")


#Create a loop to check which poly have a smaller MSE

train_mses =[]
cv_mses=[]
models = []
scalers = []

for degree in range(1,11):
    #Add poly feature to the training set
    poly = PolynomialFeatures(degree,include_bias=False)
    x_train_mapped = poly.fit_transform(x_train)

    #Scale the training set
    scaler_poly = StandardScaler()
    x_train_mapped_scaled = scaler_poly.fit_transform(x_train_mapped)
    scalers.append(scaler_poly)

    #Create and training the model
    model = LinearRegression()
    model.fit(x_train_mapped_scaled,y_train)
    models.append(model)

    #Compute the training MSE
    y_hat = model.predict(x_train_mapped_scaled)
    train_MSE = mean_squared_error(y_train,y_hat)/2
    train_mses.append(train_MSE)

    #Compute the cv MSE
    poly = PolynomialFeatures(degree,include_bias=False)
    x_cv_mapped = poly.fit_transform(x_cv)
    x_cv_mapped_scaled = scaler_poly.transform(x_cv_mapped)
    y_hat = model.predict(x_cv_mapped_scaled)
    cv_MSE = mean_squared_error(y_cv,y_hat)/2
    cv_mses.append(cv_MSE)

#Plot the result

print(cv_mses)
degrees = range(1,11)
plt.plot(degrees,train_mses, marker='o', c='r', label='training MSEs') 
plt.plot(degrees,cv_mses,marker='o',c='b',label="cv mse")
plt.title("degree of polynomial vs. train and CV MSEs")
plt.xlabel("degree")
plt.ylabel("MSE")
plt.legend()
plt.show()

#choosing the best model
degree = np.argmin(cv_mses)+1
print(f"Lowest CV MSE is found in the model with degree={degree}")
 
##Compute the test MSE
poly = PolynomialFeatures(degree,include_bias=False)
x_test_mapped = poly.fit_transform(x_test)

x_test_mapped_scaled=scalers[degree-1].transform(x_test_mapped)

y_hat = models[degree-1].predict(x_test_mapped_scaled)

test_mse = mean_squared_error(y_test,y_hat)/2

print(f"the test for {degree} poly is {test_mse:0.2f}")


##--------Neural Network----------------

#mapping all train cv test set
degree = 1
poly = PolynomialFeatures(degree,include_bias=False)
x_train_mapped = poly.fit_transform(x_train)
x_cv_mapped = poly.transform(x_cv)
x_test_mapped = poly.transform(x_test)

#scaling the origin data
scaler_poly = StandardScaler()
x_train_mapped_scaled = scaler_poly.fit_transform(x_train_mapped)
x_cv_mapped_scaled = scaler_poly.transform(x_cv_mapped)
x_test_mapped_scaled = scaler_poly.transform(x_test_mapped)

#build the model
models = []

tf.random.set_seed(20)

model_1 = Sequential(
    [
        Dense(25,activation='relu'),
        Dense(15,activation='relu'),
        Dense(1,activation='linear')
    ],
    name="model_1"
)


model_2 = Sequential(
    [
        Dense(20,activation='relu'),
        Dense(12,activation='relu'),
        Dense(12,activation='relu'),
        Dense(20,activation='relu'),
        Dense(1,activation='linear')
    ],
    name="model_2"
)

model_3 = Sequential(
    [
        Dense(32,activation='relu'),
        Dense(16,activation='relu'),
        Dense(8,activation='relu'),
        Dense(4,activation='relu'),
        Dense(12,activation='relu'),
        Dense(1,activation='linear')
    ],
    name="model_3"
)

models.append(model_1)
models.append(model_2)
models.append(model_3)

models = utils.build_models()
#training the data and get the mse

train_mses =[]
cv_mses =[]
test_mse = 0

for model in models:

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1)
    )
    
    print(f"Training {model.name}...")

    model.fit(
        x_train_mapped_scaled,y_train,
        epochs= 300,
        verbose = 0
    )

    print("Done!")

    y_hat = model.predict(x_train_mapped_scaled)
    train_mse = mean_squared_error(y_train,y_hat)/2
    train_mses.append(train_mse)

    y_hat = model.predict(x_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv,y_hat)/2
    cv_mses.append(cv_mse)

print(train_mses)
print(cv_mses)

#plot the result 
#utils.plot_train_cv_mses(range(1,4),train_mses,cv_mses,"train and cv")

print("RESULTS:")
for model_num in range(len(train_mses)):
    print(
        f"Model {model_num+1}: Training MSE: {train_mses[model_num]:.2f}, " +
        f"CV MSE: {cv_mses[model_num]:.2f}"
        )

# Select the model with the lowest CV MSE
model_num = 3

# Compute the test MSE
yhat = models[model_num-1].predict(x_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

print(f"Selected Model: {model_num}")
print(f"Training MSE: {train_mses[model_num-1]:.2f}")
print(f"Cross Validation MSE: {cv_mses[model_num-1]:.2f}")
print(f"Test MSE: {test_mse:.2f}")
"""

#--------Classification--------------

data = np.loadtxt('/home/klein/Git_J/AI/Machine-Learning-Specialization-Coursera/C2 - Advanced Learning Algorithms/week3/optional labs/data/data_w3_ex2.csv',delimiter=',')

x_bc = data[:,:-1]
y_bc = data[:,-1]

y_bc = np.expand_dims(y_bc,axis=1)

#plot the data
"""
plt.rcParams['figure.figsize']=(10,10)
plt.rcParams["lines.markersize"]=8
for i in range(len(y_bc)):
    marker = 'x' if y_bc[i]==1 else 'o'
    c = 'r' if y_bc[i]==1 else 'b'
    plt.scatter(x_bc[i,0],x_bc[i,1],marker=marker,c=c)
plt.title('x1 vs x2')
plt.xlabel("x1"); 
plt.ylabel("x2"); 
plt.show()
"""
#Split and Get 60% trainingset 20% cvset 20% testset
x_bc_train,x_,y_bc_train,y_ = train_test_split(x_bc,y_bc,test_size=0.4,random_state=1)
x_bc_test,x_bc_cv,y_bc_test,y_bc_cv = train_test_split(x_,y_,test_size=0.5,random_state=1)
del x_,y_

print(f"the shape of the training set (input) is: {x_bc_train.shape}")
print(f"the shape of the training set (target) is: {y_bc_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_bc_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_bc_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_bc_test.shape}")
print(f"the shape of the test set (target) is: {y_bc_test.shape}")

#Evaluate the error for classification models
"""
probabilities = np.array([0.2, 0.6, 0.7, 0.3, 0.8])
predictions = np.where(probabilities >= 0.5, 1, 0)
ground_truth = np.array([1, 1, 1, 1, 1])
misclassified = 0
num_predictions = len(predictions)
for i in range(num_predictions):
    
    # Check if it matches the ground truth
    if predictions[i] != ground_truth[i]:
        
        # Add one to the counter if the prediction is wrong
        misclassified += 1

# Compute the fraction of the data that the model misclassified
fraction_error = misclassified/num_predictions
"""

#Build and train data
train_errors = []
cv_errors = []


models_bc = utils.build_models()

for model in models_bc:
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
    )
    
    print(f"Training {model.name}...")

    model.fit(
        x_bc_train,y_bc_train,
        epochs=200,
        verbose=0
    )

    print("Done！")

    threshold = 0.5

    y_hat = model.predict(x_bc_train)
    y_hat = tf.math.sigmoid(y_hat)
    y_hat = np.where(y_hat>=threshold,1,0)
    train_error = np.mean(y_hat !=y_bc_train)
    train_errors.append(train_error)

    y_hat = model.predict(x_bc_cv)
    y_hat = tf.math.sigmoid(y_hat)
    y_hat = np.where(y_hat>=threshold,1,0)
    cv_error = np.mean(y_hat != y_bc_cv)
    cv_errors.append(cv_error)

for model_num in range(len(train_errors)):
    print(
        f"Model {model_num+1}: Training Set Classification Error；{train_errors[model_num]:.5f},"
        f"CV Set Classification Error:{cv_errors[model_num]:.5f}"
    )

#Get the test_error for the last model
threshold = 0.5
y_hat = models_bc[np.argmin(cv_errors)].predict(x_bc_test)
y_hat =tf.math.sigmoid(y_hat)
y_hat =np.where(y_hat>=threshold,1,0)
test_error = np.mean(y_hat != y_bc_test)
print (f"The most correct models is {models_bc[np.argmin(cv_errors)].name},and the test_error is {test_error}")