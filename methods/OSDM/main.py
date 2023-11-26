# @author Jay Kumar

from Document import Document
from Model import Model
import json
import time
import sys

# Get parent working directory
from pathlib import Path
parent = Path.cwd().parents[1]
parent


def printExecutionTime(runtime, str=""):
    print(str+ " time elapsed: {:.2f}s".format(runtime))

def outputFileNameFormatter(resultDir, dataset, outputPrefix, ALPHA, BETA, LAMDA, decay):
    output = ""
    if decay == True:
        output = resultDir + "/" + dataset + outputPrefix + "_ALPHA" + str(ALPHA) + "_BETA" + str(
            BETA) + "_LAMDA" + str(LAMDA) + ".txt"
    else:
        output = resultDir + "/" + dataset + outputPrefix + "_ALPHA" + str(ALPHA) + "_BETA" + str(BETA) + ".txt"
    print("ALHA " + str(ALPHA) + " -  BETA " + str(BETA))
    return output



dataDir = parent / "data/"
outputPath = "results/"

# dataset = "News"
#dataset = "Tweets"
# dataset = "reuters21578"

datasets = ["news_small1", "news_small2", "news_small3", \
            "tweets_small1", "tweets_small2", "tweets_small3", \
            "stack_small1", "stack_small2", "stack_small3", \
            "quora_small1", "quora_small2", "quora_small3", \
            ]

for dataset in datasets:

    startTime = time.time()
    
    LAMDA = 0.000006
    alphas = [0.005]
    betas =  [0.005]

    decay = True
    applyICF = True
    applyCWW = True
    start_index = 0

    outputPrefix = ""
    if applyICF:
        outputPrefix = outputPrefix+"_ICF"
    if applyCWW:
        outputPrefix = outputPrefix + "_CWW"
    start_time = time.time()
    print("Dataset: ",dataset," , Decay:", decay, " , ICF = ", applyICF, " , CWW = ", applyCWW)
    listOfObjects = []
    with open(dataDir / dataset) as input:  #load all the objects in memory
        line = input.readline()
        while line:
            obj = json.loads(line)  # a line is a document represented in JSON
            listOfObjects.append(obj)
            line = input.readline()

    indexOfAlpha = -1
    indexOfBeta = -1
    for a in alphas:
        indexOfAlpha += 1
        for b in betas:
            indexOfBeta += 1
            if indexOfAlpha!=indexOfBeta:
                continue
            if a == 0.0 or b == 0.0:
                continue
            ALPHA = a
            BETA = b

            output = outputPath + "/" + dataset + ".txt"

            model = Model(ALPHA, BETA, LAMDA, applyDecay=decay, applyICF = applyICF, applyCWW=applyCWW)
            iter = 1
            for obj in listOfObjects:
                document = Document(obj, model.word_wid_map, model.wid_word_map,
                                    model.wid_docId, model.word_counter)  # creating a document object which will spilt the text and update wordToIdMap, wordList
                model.processDocument(document)
                iter += 1


            endTime = time.time()
            runtime = endTime - startTime
            printExecutionTime(runtime)
            
            # Printing Clusters into File
            f = open(output, "w")
            st = "0 " +str(runtime) + "\n"
            f.write(st)
            for d in model.docIdClusId:
                st = ""+str(d)+" "+str(model.docIdClusId[d])+" \n"
                f.write(st)
            for d in model.deletedDocIdClusId:
                st = ""+str(d)+" "+str(model.deletedDocIdClusId[d])+" \n"
                f.write(st)
            f.close()
            print(output)
        indexOfBeta = -1
        # end of beta loop
    #end of alpha loop
       

