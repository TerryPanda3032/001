import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import*
from keras import layers
import numpy as np
from keras import activations
####################################
print(" import==100%")
Axis_x=np.linspace(0, 100,60)
Axis_y=0
Axis_y=3*Axis_x + 6 + np.random.randn(60) * 4
print(Axis_x,Axis_y)
####################################
model=keras.Sequential()
model.add(layers.Dense(512,input_dim=1,activation="relu"))
model.add(layers.Dense(128,input_dim=512,activation="relu"))
model.add(layers.Dense(128,input_dim=128))
model.add(layers.Dense(128,input_dim=128))
model.add(layers.Dense(33,input_dim=128))
model.add(layers.Dense(12,input_dim=33))
model.add(layers.Dense(1,input_dim=12))
model.summary()
######################################
model.compile(optimizer="adam",loss="mse")
#######################################
model.fit(Axis_x,Axis_y,epochs=1700)
#####################################
x=model.predict(Axis_x)
plt.scatter(Axis_x,Axis_y)
plt.scatter(Axis_x,x)
plt.show()




