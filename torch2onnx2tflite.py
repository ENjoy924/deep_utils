import numpy as np
import torch
import torchvision.transforms
from torchvision.models import resnet18
from PIL import Image
from torchvision.models import _utils
import onnx
import tensorflow as tf
from tensorflow.keras.applications.resnet import decode_predictions
from onnx_tf.backend import prepare
'''
    pytorch模型->onnx->tflite
'''
# gpus = tf.config.experimental.list_physical_devices()
# for gpu in gpus:
#     if gpu.device_type == 'gpu':
#         tf.config.experimental.set_memory_growth(gpu,True)
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# session = tf.compat.v1.Session(config=config)
#
#
model = resnet18(pretrained=True)
print(model.state_dict().keys())
print(np.array(model.conv1.weight.data).shape)
# for param in model.parameters():
#     print(param)
img_path = '1.jpg'
img = Image.open(img_path)
img = img.resize([224,224])
img = np.array(img,dtype=np.float32)
img = np.expand_dims(img,axis=0)
img = np.transpose(img,axes=[0,3,1,2])
img_tensor = torch.from_numpy(img)
# img_tensor = img_tensor.permute([0,3,1,2]).float()

model.eval()
result = model(img_tensor)
print(np.argmax(result[0].data))

torch.onnx.export(model,torch.rand(size=(1,3,224,224)),'model.onnx',verbose=False,input_names=['input_0'],output_names=['output_0'])

onnx_model = onnx.load_model('model.onnx')

tf_exp = prepare(onnx_model)
tf_exp.export_graph('save_model')

converter = tf.lite.TFLiteConverter.from_saved_model('save_model')
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

with open('model.tflite','wb') as g:
    g.write(tflite_model)

model = tf.lite.Interpreter('model.tflite')
input_tensor = model.get_tensor_details()
output_tensor = model.get_output_details()
print(input_tensor)
print(output_tensor)
input_data = input_tensor[0]['index']
output_data = output_tensor[0]['index']
model.allocate_tensors()
model.set_tensor(input_data,img)
model.invoke()
result = model.get_tensor(output_data)
print(np.argmax(result[0]))




