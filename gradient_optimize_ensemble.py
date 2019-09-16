#!/bin/env python
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
from keras import activations
from importance_utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.allow_soft_placement = True
sess = tf.Session(config=config)
set_session(sess)

print(device_lib.list_local_devices())
parser = argparse.ArgumentParser()
parser.add_argument('testfasta')
parser.add_argument('model',help="model folder")
parser.add_argument('ioutfile')
opts=parser.parse_args()

X = fa_to_onehot(opts.testfasta)
model_folders = [opts.model+"/"+d for d in os.listdir(opts.model) if os.path.isdir(opts.model+"/"+d)]
with open(opts.model+"/model_acc.pkl","rb") as f:
    accuracies = pickle.load(f)
for mi,model in enumerate(model_folders):
    print(model)
    grads_ed = trace_to_conv_layer(1,model+"/model.h5",0,X,0.1)
    grads_es = trace_to_conv_layer(1,model+"/model.h5",1,X,0.1)
    sum_importance_ed = np.sum(grads_ed["grads"],axis=1) #samples x filters
    sum_importance_es = np.sum(grads_es["grads"],axis=1) #samples x filters
    with open(model+"/"+opts.ioutfile+'_ed.pkl', 'wb') as handle:
        pickle.dump(sum_importance_ed, handle, protocol=2)
    with open(model+"/"+opts.ioutfile+'_es.pkl', 'wb') as handle:
        pickle.dump(sum_importance_es, handle, protocol=2)
        
