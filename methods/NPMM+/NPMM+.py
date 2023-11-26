from sklearn.datasets import fetch_20newsgroups
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import defaultdict
import gensim
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import nltk
import json
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

from scipy.special import gamma, gammaln, loggamma
from numpy import log, pi, linalg, exp, e
import random
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
import sys
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


def run_NPMM(dataset):

    startTime = time.time()

    def get_embed(questions):
        for i in range(len(questions)):
            #questions[i] = np.ones(512)
            questions[i] = encoder([questions[i]])[0].numpy()
        return questions

    def pre_processData(newsgroups_train):
        tokenizer = RegexpTokenizer(r'\w+')
        for i in range(len(newsgroups_train)):
            newsgroups_train[i] = (newsgroups_train[i]).lower()
            newsgroups_train[i] = tokenizer.tokenize(newsgroups_train[i])
        newsgroups_train = [[token for token in doc if not token.isdigit()] for doc in newsgroups_train]
        lemmatizer = WordNetLemmatizer()
        newsgroups_train = [[lemmatizer.lemmatize(token) for token in doc] for doc in newsgroups_train]
        return newsgroups_train

    def remove_stopwords(documents,news_labels):
        temp_corpus = {}
        temp_label = {}
        stop_en = stopwords.words('english')
        i = 0
        for index, words in enumerate(documents):
            rwords=[]
            for word in words:
                if word not in stop_en:
                    rwords.append(word)
            if rwords:
                temp_corpus[i] = rwords
                temp_label[i] = news_labels[index]
                i = i+1
            else:
                temp_corpus[i] = words
                temp_label[i] = news_labels[index]
                i = i+1
        return temp_corpus,temp_label

    def process_wordvectors(vocab,vectors,documents,news_labels):
        useable_vocab = 0
        unusable_vocab = 0
        temp_corpus = {}
        temp_labels = {}
        temp_unvecs = {}
        i=0
        for index in range(len(documents)):
            filter_word = []
            words = documents[index]
            temp_unvecs[i] = {}
            for word in words:
                try:
                    vectors[word]
                    vocab.add(word)
                    filter_word.append(word)
                    useable_vocab += 1
                except:
                    unusable_vocab += 1
                    if word in temp_unvecs[i].keys():
                        temp_unvecs[i][word] +=1
                    else:
                         temp_unvecs[i][word] = 1
                    continue
            if filter_word and len(words)>0 :
                temp_corpus[i] = filter_word
                temp_labels[i] = news_labels[index]

                i= i+1
            else:
                temp_corpus[i] = 'unk'
                temp_labels[i] = news_labels[index]
                i= i+1
        print("There are {0} words that could be convereted to word vectors in your corpus \n" \
              "There are {1} words that could NOT be converted to word vectors".format(useable_vocab, unusable_vocab))
        print("doc num: ",i)
        print("label num: ",i)
        return temp_corpus,temp_labels,temp_unvecs

    wordvec_filepath = parent / "Methods/glove.6B.50d.txt"

    docids = list()
    newsgroups_train = list()
    news_labels = list()
    supwords_train = list()
    file_path= parent / 'Data' / dataset
    questions = []

    with open(file_path) as fp:
        lines = fp.read().split("\n")
        for line in lines:
            if line:
                docid = json.loads(line)["Id"]
                text = json.loads(line)["textCleaned"].strip()
                label = json.loads(line)["clusterNo"]
                docids.append(docid)
                newsgroups_train.append(text)
                news_labels.append(label)
                questions.append(text)
    fp.close()

    corpus = pre_processData(newsgroups_train)
    embed = get_embed(questions)
    corpus,news_labels = remove_stopwords(corpus,news_labels)
    vectors = gensim.models.KeyedVectors.load_word2vec_format(fname=wordvec_filepath, binary=False)
    vocab = set([])
    corpus,news_labels,supwords_train = process_wordvectors(vocab,vectors,corpus,news_labels)
    print(len(news_labels))
    print(len(corpus))

    doc_words = {}
    for docID in range(len(corpus)):
        doc_words[docID] = {}
        words = corpus[docID]
        for word in words:
            if word not in doc_words[docID].keys():
                doc_words[docID][word] = 0
            doc_words[docID][word] += 1

    alpha = 0.03
    beta = 0.03
    docID_assign_z = {}
    m_z = {}
    n_z = {}
    n_w = {}
    Topics = []
    docID = 0
    initial_z = 0
    V = set()
    D = set()
    beta_topic_sum = {}
    beta_topic_v = {}
    gamma = 0.03
    gammaS = 0.0000001

    D.add(docID)
    docID_assign_z[docID] = initial_z
    words = corpus[docID]
    if initial_z not in m_z.keys():
        m_z[initial_z] = set()
    m_z[initial_z].add(docID)
    for word in words:
        if initial_z not in n_w.keys():
            n_w[initial_z] = 0
        if initial_z not in n_z.keys():
            n_z[initial_z] = {}
        if word not in n_z[initial_z].keys():
            n_z[initial_z][word] = 0
        n_z[initial_z][word] += 1
        n_w[initial_z] += 1
        V.add(word)


    def sum_topic_word():
        global_important_word = []
        for k in topic_keyword.keys():
            global_important_word.extend(list(topic_keyword[k]))
        x = None
        if len(global_important_word) != 0:
            vec_dim = 0
            for word in global_important_word:
                vec_dim += 1
                if x is not None:
                    x = np.row_stack((x, vectors[word]))
                else:
                    x = vectors[word]
            global_v_bar_k[0] = None
            if vec_dim > 1:
                global_v_bar_k[0] = np.mean(x, axis=0)[:, None]
            else:
                x = x[:, None]
                global_v_bar_k[0] = x
            x = x.T
            global_kappa_k[0] = kappa0 + vec_dim
            global_N[0] = vec_dim
            global_C_k[0] = (x - global_v_bar_k[0]).dot((x - global_v_bar_k[0]).T)
            global_mu_k[0] = (kappa0 * mu0 + global_N[0] * global_v_bar_k[0]) / global_kappa_k[0]
            global_psi_k[0] = psi + global_C_k[0] + (kappa0 * global_N[0] / global_kappa_k[0]) * (
                (global_v_bar_k[0] - mu0).T.dot(global_v_bar_k[0] - mu0))
            global_nu_k[0] = nu0 + vec_dim
            shakage_v = vec_dim + 1
            global_cov_k[0] = global_psi_k[0] / (shakage_v)
            global_inv_cov_k[0] = np.linalg.inv(global_cov_k[0])
        else:
            shakage_v = 20
            global_cov_k[0] = psi / (shakage_v)
            global_mu_k[0] = mu0
            global_inv_cov_k[0] = np.linalg.inv(global_cov_k[0])
        return global_important_word


    def sampleBetaAssignment(k, word, iter, total_iter, max_word_prob):
        if beta_topic_v[k][word] == 1:
            beta_topic_sum[k] -= 1
        pBetaAllOthers = beta_topic_sum[k]

        log_true = (n_z[k][word] / n_w[k]) / max_word_prob
        log_false = 1 - log_true

        log_p = []
        if log_false < 0 or log_true < 0:
            a = 1 / 0
        log_p.append(log_false)
        log_p.append(log_true)

        sum_pro = sum(log_p)
        normalized_posterior = [i / sum_pro for i in log_p]
        update_k = np.random.choice(2, 1, p=normalized_posterior)[0]
        if iter == total_iter - 1:
            update_k = 0
            if log_false < log_true:
                update_k = 1
        if update_k == 1:
            beta_topic_v[k][word] = 1
            beta_topic_sum[k] += 1
            topic_keyword[k].add(word)
        else:
            beta_topic_v[k][word] = 0
            topic_keyword[k].discard(word)

    compara_batch = [len(corpus)]

    for i_batch in compara_batch:
        start = 0
        end = 0
        total_batch = None
        if len(corpus) % i_batch == 0:
            total_batch = int(len(corpus) / i_batch)
        else:
            total_batch = int(len(corpus) / i_batch) + 1

        for batch in range(total_batch):
            docID_assign_z = {}
            m_z = {}
            n_z = {}
            n_w = {}
            Topics = []
            V = set()
            D = set()
            beta_topic_v = {}
            beta_topic_sum = {}
            topic_keyword = {}

            global_v_bar_k = {}
            global_C_k = {}
            global_mu_k = {}
            global_psi_k = {}
            global_nu_k = {}
            global_kappa_k = {}
            global_cov_k = {}
            global_N = {}
            global_inv_cov_k = {}
            global_cov_det_k = {}
            kappa0 = 0.01
            dim = 50
            vec_x = 1.0
            nu0 = dim
            psi = np.eye(dim)
            mu0 = np.array([vec_x for i in range(dim)])[:, None]
            alpha = 0.03
            gamma = 30
            gammaS = 0.03
            global_word = {}
            global_important_word = None

            end = i_batch * (batch + 1)
            if end > len(corpus):
                end = len(corpus)
            total_iter = 1
            for iter in range(total_iter):
                print("iter ", iter, " total K ", len(Topics))
                mean_emb = np.ones(512)
                for docID in range(start, end):
                    words = corpus[docID]
                    D.discard(docID)
                    if docID in docID_assign_z.keys():
                        before_k = docID_assign_z[docID]
                        m_z[before_k].discard(docID)
                        for word in words:
                            global_word[word] -= 1
                            n_z[before_k][word] -= 1
                            n_w[before_k] -= 1
                        k = before_k
                        max_word_prob = n_w[k]
                        if max_word_prob != 0:
                            max_word_prob = max(n_z[k].values()) / max_word_prob

                        for word in n_z[k].keys():
                            if word not in beta_topic_v[k].keys():
                                beta_topic_v[k][word] = 0
                            if n_z[k][word] > 0:
                                sampleBetaAssignment(k, word, iter, total_iter, max_word_prob)
                            else:
                                if beta_topic_v[k][word] == 1:
                                    beta_topic_sum[k] -= 1
                                    beta_topic_v[k][word] = 0
                                    topic_keyword[k].discard(word)
                        if docID % 500 == 0:
                            global_important_word = sum_topic_word()
                    else:
                        before_k = -1

                    if len(D) == 0 and len(V) == 0:
                        choose_k = 0
                        D.add(docID)
                        docID_assign_z[docID] = choose_k
                        if choose_k not in beta_topic_v.keys():
                            beta_topic_v[choose_k] = {}
                        if choose_k not in beta_topic_sum.keys():
                            beta_topic_sum[choose_k] = 0
                        if choose_k not in m_z.keys():
                            m_z[choose_k] = set()
                        if choose_k not in topic_keyword.keys():
                            topic_keyword[choose_k] = set()
                        m_z[choose_k].add(docID)
                        for word in words:
                            if choose_k not in n_w.keys():
                                n_w[choose_k] = 0
                            if choose_k not in n_z.keys():
                                n_z[choose_k] = {}
                            if word not in n_z[choose_k].keys():
                                n_z[choose_k][word] = 0
                            if word not in beta_topic_v[choose_k].keys():
                                beta_topic_v[choose_k][word] = 0
                            if word not in global_word.keys():
                                global_word[word] = 0
                            global_word[word] += 1
                            n_z[choose_k][word] += 1
                            n_w[choose_k] += 1
                            V.add(word)
                        if choose_k not in Topics:
                            Topics.append(choose_k)
                    else:
                        log_pro = []

                        must_update_flag = 0
                        update_pro = 1
                        not_update_pro = 1
                        if_update_k = []
                        skip = 0
                        for word in words:
                            if word in global_important_word:
                                skip = 1
                                must_update_flag = 1
                                break
                        if skip == 0:
                            pro = (1 - ((cosine_similarity(embed[docID].reshape(1,-1), mean_emb.reshape(1,-1))[0][0] + 1) / 2))
                            update_pro *= pro
                            not_update_pro *= (1 - pro)

                        if must_update_flag == 0:
                            if_update_k.append(not_update_pro)
                            if_update_k.append(update_pro)
                            sum_pro = sum(if_update_k)
                            normalized_posterior = [i / sum_pro for i in if_update_k]
                            update_k = np.random.choice(2, 1, p=normalized_posterior)[0]
                
                        else:
                            update_k = 1

                        choose_k = None
                        if update_k == 1:
                            for k in Topics:
                                pro_k = len(m_z[k])
                                if pro_k == 0:
                                    log_pro.append(0)
                                else:
                                    i = 0
                                    for word in words:
                                        if word not in n_z[k].keys():
                                            n_z[k][word] = 0
                                        bias_flag = 0
                                        if word in beta_topic_v[k].keys():
                                            bias_flag = beta_topic_v[k][word]
                                        for j in range(doc_words[docID][word]):
                                            pro_k *= (n_z[k][word] + bias_flag * gamma + gammaS + j) / (
                                                        n_w[k] + beta_topic_sum[k] * gamma + len(V) * gammaS + i)
                                            i += 1
                                    log_pro.append(pro_k)
                            sum_pro = sum(log_pro)
                            normalized_posterior = [i / sum_pro for i in log_pro]
                            select_k = None
                            if iter == (total_iter - 1):
                                select_k = normalized_posterior.index(max(normalized_posterior))

                            else:
                                select_k = np.random.choice(len(Topics), 1, p=normalized_posterior)[0]
                            choose_k = Topics[select_k]
                        else:
                            choose_k = np.max(Topics) + 1

                        D.add(docID)
                        docID_assign_z[docID] = choose_k
                        if choose_k not in m_z.keys():
                            m_z[choose_k] = set()
                        m_z[choose_k].add(docID)
                        if choose_k not in beta_topic_v.keys():
                            beta_topic_v[choose_k] = {}
                        if choose_k not in beta_topic_sum.keys():
                            beta_topic_sum[choose_k] = 0
                        if choose_k not in topic_keyword.keys():
                            topic_keyword[choose_k] = set()
                        for word in words:
                            if choose_k not in n_w.keys():
                                n_w[choose_k] = 0
                            if choose_k not in n_z.keys():
                                n_z[choose_k] = {}
                            if word not in n_z[choose_k].keys():
                                n_z[choose_k][word] = 0
                            if word not in beta_topic_v[choose_k].keys():
                                beta_topic_v[choose_k][word] = 0
                            if word not in global_word.keys():
                                global_word[word] = 0
                            global_word[word] += 1
                            n_z[choose_k][word] += 1
                            n_w[choose_k] += 1
                            V.add(word)
                        if choose_k not in Topics:
                            Topics.append(choose_k)

                    count_k = []
                    for k in Topics:
                        if k in m_z.keys() and len(m_z[k]) == 0:
                            m_z.pop(k, None)
                            n_z.pop(k, None)
                            n_w.pop(k, None)
                            beta_topic_v.pop(k, None)
                            beta_topic_sum.pop(k, None)
                            beta_topic_v.pop(k, None)
                            topic_keyword.pop(k, None)
                            count_k.append(k)
                    for k in count_k:
                        Topics.remove(k)

                    k = choose_k
                    max_word_prob = n_w[k]
                    if max_word_prob != 0:
                        max_word_prob = max(n_z[k].values()) / max_word_prob

                    for word in n_z[k].keys():
                        if word not in beta_topic_v[k].keys():
                            beta_topic_v[k][word] = 0
                        if n_z[k][word] > 0:
                            sampleBetaAssignment(k, word, iter, total_iter, max_word_prob)
                        else:
                            if beta_topic_v[k][word] == 1:
                                beta_topic_sum[k] -= 1
                                beta_topic_v[k][word] = 0
                                topic_keyword[k].discard(word)
                    if docID % 100 == 0:
                        global_important_word = sum_topic_word()

                stack_emb = np.vstack((embed[docID], mean_emb))
                mean_emb = np.mean(stack_emb, axis=0)


    endTime = time.time()

    runtime = endTime - startTime
    outputPath = "results/" + dataset +".txt"

    writer = open(outputPath, 'w')
    writer.write("0 " + str(runtime) + "\n")
    for i, docid in enumerate(docids):
        documentID = docid
        cluster = docID_assign_z[i]
        writer.write(str(documentID) + " " + str(cluster) + "\n")
    writer.close()

for dataset in datasets:
    run_NPMM(dataset)
