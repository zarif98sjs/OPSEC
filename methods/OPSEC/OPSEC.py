import numpy as np
import pandas as pd
import json
import time

# Get parent working directory
from pathlib import Path
parent = Path.cwd().parents[1]
parent

import tensorflow_hub as hub
encoder = hub.load(str(parent / 'Methods/universal-sentence-encoder_4'))
from sklearn.metrics.pairwise import cosine_similarity

datasets = ["news_small1", "news_small2", "news_small3", \
            "tweets_small1", "tweets_small2", "tweets_small3", \
            "stack_small1", "stack_small2", "stack_small3", \
            "quora_small1", "quora_small2", "quora_small3", \
            ]

def run_opsec(dataset):
    file_path= parent / 'Data' / dataset

    docids = list()
    news_labels = list()
    questions = []

    with open(file_path) as fp:
        lines = fp.read().split("\n")
        for line in lines:
            if line:
                docid = json.loads(line)["Id"]
                label = json.loads(line)["clusterNo"]
                text = json.loads(line)["textCleaned"]
                docids.append(docid)
                news_labels.append(label)
                questions.append(text)
                
    startTime = time.time()   
    clus = {}
    cluid = 0
    alpha = 0.35
    assignid = []
    for i, q in enumerate(questions):
        if clus:
            maxsim = -np.inf
            addclu = -np.inf
            emb = encoder([q])
            for clu in clus:
                sim = np.inner(emb, clus[clu]['cent']) 
                if sim > maxsim:
                    maxsim = sim
                    addclu = clu
            if (maxsim > alpha):
                clus[addclu]['docid'].append(docids[i])
                clus[addclu]['doc'].append(q)
                clus[addclu]['cent'] = np.mean(np.vstack([emb, clus[addclu]['cent']]), axis=0)
                assignid.append(addclu)
            else:
                clus[cluid] = {'docid': [docid], 'doc': [q], 'cent': encoder([q])}
                assignid.append(cluid)
                cluid+=1   
        else:
            clus[cluid] = {'docid': [docid], 'doc': [q], 'cent': encoder([q])}
            assignid.append(cluid)
            cluid+=1
    endTime = time.time()
    runtime = endTime - startTime
    
    outputPath = 'results/' + dataset + '.txt'

    writer = open(outputPath, 'w')
    writer.write("0 " + str(runtime) + "\n")
    for i, docid in enumerate(docids):
        documentID = docid
        cluster = assignid[i]
        writer.write(str(documentID) + " " + str(cluster) + "\n")
    writer.close()

for dataset in datasets:
    run_opsec(dataset)
