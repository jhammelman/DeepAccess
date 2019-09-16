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
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.allow_soft_placement = True
sess = tf.Session(config=config)
set_session(sess)

print(device_lib.list_local_devices())
parser = argparse.ArgumentParser()
parser.add_argument('testfasta')
parser.add_argument('model',help="model folder")
parser.add_argument('ioutfile')
parser.add_argument('outfile')
opts=parser.parse_args()

X = fa_to_onehot(opts.testfasta)
model_folders = [opts.model+"/"+d for d in os.listdir(opts.model) if os.path.isdir(opts.model+"/"+d)]
with open(opts.model+"/model_acc.pkl","rb") as f:
    accuracies = pickle.load(f)
total_grads_ed = []
total_grads_es = []
for mi,model in enumerate(model_folders):
    print(model)
    grads_ed = saliency(0,model+"/model.h5",0,X,30)*X
    grads_es = saliency(0,model+"/model.h5",1,X,30)*X
    # grads are a X size matrix with importance scores for each
    # sequence, for each position in the sequence
    total_grads_ed.append(grads_ed)
    total_grads_es.append(grads_es)
    with open(model+"/"+opts.ioutfile+'_tp2.pkl', 'wb') as handle:
        pickle.dump(grads_ed, handle, protocol=2)
    with open(model+"/"+opts.ioutfile+'_tp1.pkl', 'wb') as handle:
        pickle.dump(grads_es, handle, protocol=2)
        
saliency_ed = np.zeros(total_grads_ed[0].shape)

for mi,model in enumerate(model_folders):    
    saliency_ed += accuracies[model]*total_grads_ed[mi]
saliency_ed = saliency_ed/sum(accuracies.values())
with open(opts.outfile+'_tp1.pkl','wb') as handle:
    pickle.dump(saliency_ed,handle,protocol=2)

saliency_es = np.zeros(total_grads_es[0].shape)
for mi,model in enumerate(model_folders):    
    saliency_es += accuracies[model]*total_grads_es[mi]
saliency_es = saliency_es/sum(accuracies.values())
with open(opts.outfile+'_tp2.pkl','wb') as handle:
    pickle.dump(saliency_es,handle,protocol=2)


