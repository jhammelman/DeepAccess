import os
import numpy as np 
import argparse
from ensemble_utils import *
from CNN import *
import argparse
import keras
import pickle
from tensorflow.python.client import device_lib
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.allow_soft_placement = True
sess = tf.Session(config=config)
set_session(sess)

print(device_lib.list_local_devices())

#exit()
parser = argparse.ArgumentParser()
parser.add_argument('testfasta')
parser.add_argument('model',help="model folder")
parser.add_argument('outfile')
parser.add_argument('-ioutfile','--ioutfile',default=None)
opts=parser.parse_args()
X = fa_to_onehot(opts.testfasta)
model_folders = [opts.model+"/"+d for d in os.listdir(opts.model) if os.path.isdir(opts.model+"/"+d)]

with open(opts.model+"/model_acc.pkl","rb") as f:
    accuracies = pickle.load(f)

print(X.shape)
total_pred = []
for mi,model in enumerate(model_folders):
    cnn = keras.models.load_model(model+"/model.h5")
    print(cnn.summary())
    pred=cnn.predict(X)
    total_pred.append(pred)
    if opts.ioutfile != None:
        np.savetxt(model+"/"+opts.ioutfile,pred)
    del cnn

pred_mat = np.zeros(total_pred[0].shape)
print(accuracies)
for mi,model in enumerate(model_folders):    
    pred_mat += accuracies[model]*total_pred[mi]
pred_mat = pred_mat/sum(accuracies.values())
np.savetxt(opts.outfile,pred_mat)
