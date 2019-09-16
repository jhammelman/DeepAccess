import os
import numpy as np 
import argparse
from ensemble_utils import *
from CNN import *
import argparse
import pickle
import keras
from tensorflow.python.client import device_lib
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from numpy.random import seed
from tensorflow import set_random_seed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.allow_soft_placement = True
sess = tf.Session(config=config)
set_session(sess)

print(device_lib.list_local_devices())

parser = argparse.ArgumentParser()
parser.add_argument('trainfasta')
parser.add_argument('trainact')
parser.add_argument('outfolder')
opts=parser.parse_args()

models = [['conv','globalpool'],
          ['conv','conv','globalpool'],
          ['conv','conv','globalpool'],
          ['conv','conv','globalpool'],
          ['conv','conv','globalpool'],
          ['conv','conv','globalpool'],
          ['conv','conv','globalpool'],
          ['conv','conv','globalpool'],
          ['conv','conv','globalpool'],
          ['conv','conv','globalpool']
]
filters =[None,
          (200,8),
          (200,16),
          (200,32),
          (400,8),
          (400,16),
          (400,32),
          (600,8),
          (600,16),
          (600,32)]
ensure_dir(opts.outfolder)

X = fa_to_onehot(opts.trainfasta)
y = act_to_class(opts.trainact)
shuffled = np.random.permutation(X.shape[0])
X = X[shuffled,:]
y = y[shuffled,:]
ntrain = int(X.shape[0]*0.9)
X_train = X[:ntrain,:]
y_train = y[:ntrain,:]
X_val = X[ntrain:,:]
y_val = y[ntrain:,:]
sample_weights = np.ones((X_train.shape[0],))
accs = {}
y_avg = np.zeros(y_train.shape)
for mi,model in enumerate(models):
    seed(mi)
    set_random_seed(mi)
    model_folder = opts.outfolder + "/" + "_".join(model) + "_" + str(mi)
    ensure_dir(model_folder)
    if filters[mi] != None:
        new_cnn = CNN(model,X_train.shape[1:],y_train.shape[1],
                      conv_filter_number=filters[mi][0],
                      conv_filter_size=filters[mi][1])
    else:
        new_cnn = CNN(model,X_train.shape[1:],y_train.shape[1])
         
    history = new_cnn.train(X_train,y_train,sample_weights)
    loss,val_acc = new_cnn.model.evaluate(X_val,y_val)
    loss,train_acc = new_cnn.model.evaluate(X_train,y_train)
    
    accs[model_folder] = val_acc
    with open(model_folder+'/model_summary.txt','w') as f:
        f.write(new_cnn.model.to_yaml())
        f.write("\nTraining Accuracy: "+str(train_acc)+"\n")        
        f.write("\nValidation Accuracy: "+str(val_acc)+"\n")
        
    new_cnn.save(model_folder+'/model.h5')
    np.save(model_folder+'/history.npy',history.history)
    np.save(model_folder+'/sample_weights.npy',sample_weights)
    sample_weights += new_cnn.error(X_train,y_train)
    print(sample_weights.shape)
    del new_cnn.model
    
with open(opts.outfolder+'/model_acc.pkl','wb') as f:
    pickle.dump(accs,f)
