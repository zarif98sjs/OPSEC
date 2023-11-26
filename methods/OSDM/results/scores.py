# Calculate the ground truth label
import pandas as pd
import json
import os 
from pathlib import Path

# Get parent working directory
cwd = Path.cwd()
cwd

import os 
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment

parent = Path.cwd().parents[2]

# Define average clustering accuracy

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = np.asarray(linear_assignment(w.max() - w))
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

measures = { 'NMI_scores': normalized_mutual_info_score, 'AMI_scores': adjusted_mutual_info_score, \
            'homo_scores': homogeneity_score, 'comp_scores': completeness_score, 'rand_scores': rand_score, \
            'adjusted_rand_scores': adjusted_rand_score, 'accu_scores': cluster_acc }

results = { 'news_small': {},'tweets_small': {}, 'stack_small': {},  'quora_small': {} }

for dataset in results:
    for measure in measures:
        results[dataset][measure] = []
        results[dataset]['times'] = []
    for i in range(1,4):
        run = dataset + str(i)
        df = pd.read_pickle(parent / ('Data/' + run + '.pkl'))
        rundata = pd.read_csv(run + '.txt', skiprows=0, delimiter=r'\s+', names=['id', 'cluster'])
        time = rundata.iloc[0][1]
        rundata = rundata[1:]
        rundata.reset_index(drop=True, inplace=True)
        rundata['cluster'] = rundata['cluster'].astype(int)
        df[run] = rundata['cluster']
        df[['textCleaned', run]]
        df['matchCount'] = df.groupby('clusterNo')['clusterNo'].transform('count')
        df = df[df['matchCount']>1].copy()
        cluster_values = df['clusterNo'].unique()
        cluster_No_new = [i for i in range(1,len(cluster_values)+1)]
        df['clusterNo'] = df['clusterNo'].replace(to_replace = cluster_values, value = cluster_No_new)
        cluster_ground_truth = list(df['clusterNo'])
        cluster_predicted = list(df[run])
        
        for measure in measures:
            results[dataset][measure].append(measures[measure](cluster_ground_truth, cluster_predicted))   
        results[dataset]['times'].append(time)

with open('scores.txt', 'w') as f:
    for measure in measures:
            print(measure, end= " ", file=f)
            for dataset in results:
                print(' & ' + str(np.round(np.mean(results[dataset][measure]), 3)) + ' +/- ' + str(np.round(np.std(results[dataset][measure]), 3)), end="", file=f)
            print(file=f)
    print('times', end= " ", file=f)
    for dataset in results:
        print(' & ' + str(np.round(np.mean(results[dataset]['times']), 3)) + ' +/- ' + str(np.round(np.std(results[dataset]['times']), 3)), end="", file=f)
