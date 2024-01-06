from tensorflow.keras import layers, models, Model, Sequential
import tensorflow as tf

def AlexNet_v1(im_height=224, im_width=224, num_classes=1000):
    # tensorflow中的通道顺序是NHWC
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  # [None, 224, 224, 3]
    # x = layers.Conv2D()
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)  # [None, 227, 227, 3]
    # (227 - 11 + 2*0) / 4 + 1 = 55  -> [None, 55, 55, 48]
    x = layers.Conv2D(filters=48, kernel_size=11, strides=4, activation="relu")(x)  
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)  # [None, 27, 27, 48]
    # 当stride=1且padding=same时， 表示输出尺寸与输入尺寸相同 ->[None, 27, 27, 128]
    x = layers.Conv2D(filters=128, kernel_size=5, padding="same", strides=1, activation="relu")(x)
    # [None, 27, 27, 128] -> [None, 13, 13, 128]
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    # stride=1, padding=same, 输出不变 ->[None, 13, 13, 192]
    x = layers.Conv2D(filters=192, kernel_size=3, padding="same", strides=1, activation="relu")(x)  
    # ->[None, 13, 13, 192]
    x = layers.Conv2D(filters=192, kernel_size=3, padding="same", strides=1, activation="relu")(x)  
    # ->[None, 13, 13, 128]
    x = layers.Conv2D(filters=128, kernel_size=3, padding="same", strides=1, activation="relu")(x)
    # ->[None, 6, 6, 128]
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = layers.Flatten()(x)  # [None, 128*6*6]
    x = layers.Dropout(0.2)(x) 
    # [None, 2048]
    x = layers.Dense(2048, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    # [None, 2048]
    x = layers.Dense(2048, activation="relu")(x)
    x = layers.Dense(num_classes)(x)

    predict = layers.Softmax()(x)
    print(predict)

    # model = models.Model(inputs=input_image, outputs=predict)
    model = models.Model(inputs=input_image, outputs=predict)
    return model


class AlexNet_v2(Model):
    def __init__(self, num_classes=1000):
        super(AlexNet_v2, self).__init__()
        self.features = Sequential([
            # [None, 224, 224, 3] -> [None, 227, 227, 3]
            layers.ZeroPadding2D(((1, 2), (1, 2))), 
            # padding="valid"表示向上取整 (227 - 11)/4 + 1=55 [None, 227, 227, 3]->[None, 55, 55, 48]
            layers.Conv2D(filters=48, kernel_size=11, strides=4, activation="relu"), 
            # [55, 55, 48] -> [None, 27, 27, 48]
            layers.MaxPool2D(pool_size=3, strides=2),
            # stride=1, padding=same, 尺寸不变 [None, 27, 27, 48] ->[None, 27, 27, 128]
            layers.Conv2D(filters=128, kernel_size=5, padding="same", activation="relu"), 
            # [None, 27, 27, 128] ->[None, 13, 13, 128]
            layers.MaxPool2D(pool_size=3, strides=2),
            # [None, 13, 13, 128] -> [None, 13, 13, 192]
            layers.Conv2D(filters=192, kernel_size=3, padding="same", activation="relu"), 
            # [None, 13, 13, 192] -> [None, 13, 13, 192]
            layers.Conv2D(filters=192, kernel_size=3, padding="same", activation="relu"), 
             # [None, 13, 13, 192] -> [None, 13, 13, 128]
            layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"), 
            # [None, 13, 13, 128] -> [None, 6, 6, 128]
            layers.MaxPool2D(pool_size=3, strides=2)])
        
        # [None, 128*6*6]
        self.flatten = layers.Flatten() 
        self.classifier = Sequential([
            layers.Dropout(0.2), 
            layers.Dense(1024, activation="relu"), 
            layers.Dropout(0.2), 
            layers.Dense(128, activation="relu"), 
            layers.Dense(num_classes), 
            layers.Softmax()  
        ])
    
    def call(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# # input = tf.random.uniform(shape=(16, 224, 32, 3))
# input =  tf.random.uniform(shape=(8, 224, 224, 3)) 
# # alexnet = AlexNet_v2(num_classes=5)
# # print(alexnet)
# # print(alexnet.call(input))
# AlexNet_v1(224, 224, 5)