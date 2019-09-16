#!/bin/env python
import numpy as np
import pickle
import argparse
from scipy.stats import norm
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('fasta')
parser.add_argument('importance')
parser.add_argument('-n','--ntop',default=25,type=int)
parser.add_argument('-k','--kmer',default=10,type=int)
parser.add_argument('-p','--pval',default=0.01,type=float,help='FDR corrected pval threshold')
opts = parser.parse_args()

seqs = [l.strip().split()[1] for l in open(opts.fasta).read().split(">")[1:]]
with open(opts.importance,'rb') as f:
    mat = pickle.load(f)

windows = np.array([np.mean(mat[i,j:(j+opts.kmer),:]) for i in range(len(seqs)) for j in range(len(seqs[0])-opts.kmer)])

mu_window = np.mean(windows)
sigma_window = np.std(windows)
significant_seqs =  {}
significant_pvals = {}
nhypothesis = windows.shape[0]
for i in range(len(seqs)):
    for j in range(len(seqs[0])-opts.kmer):
        score=np.mean(mat[i,j:(j+opts.kmer)])
        pval = norm.sf(score,
                       loc=mu_window,scale=sigma_window)
        if pval < opts.pval/(nhypothesis):
            try:
                key = significant_seqs[seqs[i][j:j+opts.kmer].upper()]
                old_pval = significant_pvals[seqs[i][j:j+opts.kmer].upper()]
                pval = min(pval,old_pval)
            except KeyError:
                significant_seqs[seqs[i][j:j+opts.kmer].upper()] = np.zeros((opts.kmer,4))
            significant_pvals[seqs[i][j:j+opts.kmer].upper()] = pval
            significant_seqs[seqs[i][j:j+opts.kmer].upper()] += mat[i,j:j+opts.kmer,:]
            
print "# num significant:",len(significant_pvals)

from Bio import pairwise2
subseqs = significant_pvals.keys()
affinity = np.zeros((len(subseqs),len(subseqs)))
for i in range(len(subseqs)-1):
    for j in range(i,len(subseqs)):
       aln = pairwise2.align.localms(subseqs[i],subseqs[j], 2, -1, -3, -1)
       if len(aln) > 0:
           score = aln[0][2]
       else:
           score = 0.01
       affinity[i,j] = max(score,0.01)
       affinity[j,i] = max(score,0.01)
bestdf=0.8
bestscore=None
sig_subtract=None
for ss in np.linspace(10,100,10):
    significance=[-np.log10(1.0/1000000+ significant_pvals[k])-ss for k in subseqs]
    labels = AffinityPropagation(damping=bestdf,affinity='precomputed',preference=significance).fit_predict(affinity)
    score = silhouette_score(affinity, labels, metric='precomputed')
    if bestscore != None and score > bestscore:
        bestscore=score
        sig_subtract=ss
    elif bestscore == None:
        bestscore=score
        sig_subtract = ss

significance=[-np.log10(1.0/1000000+ significant_pvals[k])-sig_subtract for k in subseqs]
aclust = AffinityPropagation(damping=bestdf,affinity='precomputed',preference=significance).fit(affinity)
representatives = [subseqs[i] for i in aclust.cluster_centers_indices_]
print "# affinity clustering damping:",bestdf
print "# affinity preference:",sig_subtract
print "# num clusters:",len(representatives)
print "# silhouette coefficient:",bestscore
sorted_significant = sorted([(k,significant_pvals[k]) for k in representatives],key=lambda kv:kv[1])
for key,pval in sorted_significant:
    #pval = significant_pvals[key]
    score_mat = significant_seqs[key]
    print ">",key+"\t"+str(round(pval,5))+"\t"+str(round(np.mean(score_mat),5))
    for j in range(opts.kmer):
        for n in range(4):
            print max(np.round(score_mat[j,n]*100+1),0),
        print 

        
