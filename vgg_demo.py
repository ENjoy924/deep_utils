import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices()
for gpu in gpus:
    if gpu.device_type == 'gpu':
        tf.config.experimental.set_memory_growth(gpu, True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.compat.v1.Session(config=config)

VGG16_model = tf.keras.applications.VGG16(include_top=True)
def prepocess(x):
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize(x, [224,224])
    x =tf.expand_dims(x, 0) # 扩维
    x = preprocess_input(x)
    return x

img_path='2.jpg'
img=prepocess(img_path)

Predictions = VGG16_model.predict(img)
print('Predicted:', decode_predictions(Predictions, top=3)[0])

from tensorflow.keras import models
last_conv_layer = VGG16_model.get_layer('block5_conv3')
heatmap_model =models.Model(VGG16_model.inputs, [last_conv_layer.output, VGG16_model.output])

import tensorflow.keras.backend as K
with tf.GradientTape() as gtape:
    conv_output, Predictions = heatmap_model(img)
    prob = Predictions[:, np.argmax(Predictions[0])] # 最大可能性类别的预测概率
    grads = gtape.gradient(prob, conv_output)  # 类别与卷积层的梯度 (1,14,14,512)
    pooled_grads = K.mean(grads, axis=(0,1,2)) # 特征层梯度的全局平均代表每个特征层权重

heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
heatmap = np.maximum(heatmap, 0)
max_heat = np.max(heatmap)
if max_heat == 0:
    max_heat = 1e-10
heatmap /= max_heat


import cv2
original_img=cv2.imread(img_path)
heatmap1 = cv2.resize(heatmap[0], (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)
heatmap1 = np.uint8(255*heatmap1)
heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_RAINBOW)
# plt.matshow(heatmap1, cmap='viridis')
frame_out=cv2.addWeighted(original_img,0.5,heatmap1,0.5,0)

plt.figure()
plt.imshow(frame_out)
plt.show()