import os
import numpy as np 

def ensure_dir(file_path):
    #directory = os.path.dirname(file_path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def act_to_class(act):
    y = []
    header = True
    for line in open(act):
        if header:
            header = False
            continue
        data  = line.strip().split()
        y.append([int(d) for d in data[1:]])
    return np.array(y)

def fa_to_onehot(fa):
    alpha = ['A','C','G','T']
    sequences = open(fa).read().split(">")[1:]
    seqdict = [seq.strip().split("\n")[1] for seq in sequences]
    seq_mat = []
    slen = max([len(seq) for seq in seqdict])
    for i,seqc in enumerate(seqdict):
        seq = np.zeros((slen,4))
        for j,c in enumerate(seqc.upper()):
            if c not in alpha:
                seq[j,:] = 0.25
            else:
                aind = alpha.index(c)
                seq[j,aind] = 1
        seq_mat.append(seq)
    return np.array(seq_mat)
