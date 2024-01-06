from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model
import tensorflow as tf

class LeNet(Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = Conv2D(filters=16, 
                            kernel_size=5,
                            strides=1, activation='relu')
        self.maxpool1 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.conv2 = Conv2D(filters=32, kernel_size=5, strides=1, activation='relu')
        self.maxpool2 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.flatten = Flatten()
        self.d1 = Dense(120, activation='relu')
        self.d2 = Dense(84, activation='relu')
        self.d3 = Dense(10, activation='softmax')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x
    
# input = tf.random.uniform((4, 3,32, 32))
input =  tf.random.uniform(shape=(8, 32, 32, 3)) 
# # print(input)
model = LeNet()
print(model(input))
# model(input)

