import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from lab_utils_softmax import plt_softmax
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def my_softmax(z):
    ez = np.exp(z)
    sm = ez/np.sum(ez)
    return(sm)

plt.close("all")
#plt_softmax(my_softmax)

#make dataset for example
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

#print(X_train)

model = Sequential(
    [
        Dense(25,activation="relu"),
        Dense(15,activation="relu"),
        Dense(4,activation='softmax'),
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)


model.fit(
    X_train,y_train,
    epochs=10
)

p_nonpreferred = model.predict(X_train[:3])
print(p_nonpreferred [:2])

preferred_model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'linear')   #<-- Note
    ]
)

preferred_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- Note
    optimizer=tf.keras.optimizers.Adam(0.001),
)

preferred_model.fit(
    X_train,y_train,
    epochs=10
)
p_preferred = preferred_model.predict(X_train)
print(f"two example output vectors:\n {p_preferred[:3]}")

sm_preferred = tf.nn.softmax(p_preferred).numpy()
print(f"two example output vectors:\n {sm_preferred[:2]}")

for i in range(5):
    print( f"{p_preferred[i]}, category: {np.argmax(p_preferred[i])}")