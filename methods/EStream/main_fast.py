import os
import time

from general_util import readlistWholeJsonDataSet
from evaluation import Evaluate_old
from read_pred_true_text import ReadPredTrueText
from clustering_term_online_fast import cluster_biterm
from word_vec_extractor import extractAllWordVecsPartialStemming


# Get parent working directory
from pathlib import Path
parent = Path.cwd().parents[1]

ignoreMinusOne=True
isSemantic=False

dataDir = parent / "data/"
outputPath = "results/"

#dataset= 'Tweets-T'   # 'QuoraSmallGroupRandom.txt' 'stackoverflow_javascript' 'stackoverflow_java' 'stackoverflow_python' 'stackoverflow_csharp' 'stackoverflow_php' 'stackoverflow_android' 'stackoverflow_jquery' 'stackoverflow_r' 'stackoverflow_java'  # 'stackoverflow_java'  'stackoverflow_cplus' 'stackoverflow_mysql' 'stackoverflow_large_tweets-T_news-T_suff' 'stackoverflow_large_tweets-T' #'News-T' 'NT-mstream-long1' #'Tweets-T' # 'stackoverflow_large' 'stackoverflow_large_tweets-T'

datasets = ["news_small1", "news_small2", "news_small3", \
            "tweets_small1", "tweets_small2", "tweets_small3", \
            "stack_small1", "stack_small2", "stack_small3", \
            "quora_small1", "quora_small2", "quora_small3", \
            ]

datasets = ["quora_small1", "quora_small2", "quora_small3"]

for dataset in datasets:

  t11=time.time()
    
  inputfile = dataDir / dataset

  #list_pred_true_words_index_postid=readStackOverflowDataSet(inputfile)
  list_pred_true_words_index=readlistWholeJsonDataSet(inputfile) 

  print(len(list_pred_true_words_index))

  all_words=[]
  for item in list_pred_true_words_index: 
    all_words.extend(item[2])
  all_words=list(set(all_words))

  gloveFile = "glove.6B.50d.txt"
  embedDim=50
  wordVectorsDic={}
  if isSemantic==True:
    wordVectorsDic=extractAllWordVecsPartialStemming(gloveFile, embedDim, all_words)

  c_bitermsFreqs={} 
  c_totalBiterms={}
  c_wordsFreqs={}
  c_totalWords={}
  c_txtIds={}
  c_clusterVecs={}
  txtId_txt={}
  last_txtId=0  
  max_c_id=0
  dic_clus__id={}

  dic_biterm__clusterId_Freq={}
  dic_biterm__allClusterFreq={}

  dic_biterm__clusterIds={}



  c_bitermsFreqs, c_totalBiterms, c_wordsFreqs, c_totalWords, c_txtIds, c_clusterVecs, txtId_txt,\
  last_txtId, dic_clus__id, dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq, \
  dic_biterm__clusterIds=cluster_biterm(t11, dataset, list_pred_true_words_index, c_bitermsFreqs, c_totalBiterms, c_wordsFreqs,
                                        c_totalWords, c_txtIds, c_clusterVecs, txtId_txt, last_txtId, max_c_id,
                                        wordVectorsDic, dic_clus__id, dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq,
                                        dic_biterm__clusterIds)


  t12=time.time()	  
  t_diff = t12-t11
  print("total time diff secs=",t_diff)  

  print('result for', inputfile)

  
  outputPath = "results/" + dataset + ".txt"

  with open(outputPath, "r+") as f: s = f.read(); f.seek(0); f.write("0 " + str(t_diff) + "\n" + s)

  #writer = open(outputPath, 'a')
  #writer.write("0 " + str(t_diff) + "\n")
  #writer.close()

