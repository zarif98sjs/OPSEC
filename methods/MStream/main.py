from MStream import MStream
import json
import time

# Get parent working directory
from pathlib import Path
parent = Path.cwd().parents[1]

dataDir = parent / "data/"
outputPath = "results/"

# dataset = "TREC"
# dataset = "TREC-T"
# dataset = "News"
# dataset = "News-T"


datasets = ["tweets_small1", "tweets_small2", "tweets_small3", \
            "stack_small1", "stack_small2", "stack_small3", \
            "quora_small1", "quora_small2", "quora_small3", \
            ]

#datasets = ["tweets_small1", "tweets_small2", "tweets_small3"]

timefil = "timefil"
MaxBatch = 2 # The number of saved batches + 1
AllBatchNum = 16 # The number of batches you want to devided the dataset to
alpha = 0.005
beta = 0.01
iterNum = 0
sampleNum = 1
wordsInTopicNum = 5
K = 0 

def runMStream(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, datasets, timefil, wordsInTopicNum):
    for dataset in datasets:
        mstream = MStream(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments(dataDir)
        for sampleNo in range(1, sampleNum+1):
            print("SampleNo:"+str(sampleNo))
            mstream.runMStream(sampleNo, outputPath)
        
def runMStreamF(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    for dataset in datasets:
        mstream = MStream(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments(dataDir)
        for sampleNo in range(1, sampleNum+1):
            print("SampleNo:"+str(sampleNo))
            mstream.runMStreamF(sampleNo, outputPath)


       

if __name__ == '__main__':
    runMStream(K, AllBatchNum, AllBatchNum, alpha, beta, iterNum, sampleNum, datasets, timefil, wordsInTopicNum)
    #runMStreamF(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, datasets, timefil, wordsInTopicNum)
   
