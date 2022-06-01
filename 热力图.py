import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import models
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import backend as K

gpus = tf.config.experimental.list_physical_devices()
for gpu in gpus:
    if gpu.device_type == 'gpu':
        tf.config.experimental.set_memory_growth(gpu, True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.compat.v1.Session(config=config)

vgg_model = VGG16()
# 1 取得最后一个卷积层
last_conv = vgg_model.get_layer("block5_conv3")
heap_model = models.Model(vgg_model.inputs, [last_conv.output, vgg_model.output])


# 2 取得该类别的预测得分
def process(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, (224, 224))
    img = tf.expand_dims(img, 0)
    img = preprocess_input(img)
    return img


img_path = "2.jpg"
img = process(img_path)
with tf.GradientTape() as tape:
    conv, preds = heap_model(img)
    # 3 将该类别的预测得分与最后一个卷积层求梯度
    grads = tape.gradient(preds[0][np.argmax(preds[0])], conv)
    pooled_grads = K.mean(grads, axis=[0, 1, 2])
# 4 将梯度值和最后一个卷积层相乘
heap_mat = tf.reduce_mean(tf.multiply(conv, pooled_grads), axis=-1).numpy()
heatmat = np.maximum(heap_mat, 0)
max_heat = np.max(heatmat)
if max_heat == 0:
    max_heat = 1e-10
heatmat /= max_heat
# 5 将最后一个卷积层进行通道合并
origin_img = cv2.imread(img_path)
# 6 读取原图像
# 7 将特征图进行放大
heap_mat1 = cv2.resize(heatmat[0], (origin_img.shape[1], origin_img.shape[0]), interpolation=cv2.INTER_CUBIC)
# 8 进行显示
heap_mat1 = np.uint8(heap_mat1 * 255)
heap_mat1 = cv2.applyColorMap(heap_mat1, cv2.COLORMAP_RAINBOW)
show_img = cv2.addWeighted(origin_img, 0.5, heap_mat1, 0.5, 0)
plt.figure()
plt.imshow(show_img)
plt.show()
