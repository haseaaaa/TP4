from collections import defaultdict
from tqdm import tqdm
import random
import lightgbm as lgb
import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from bsbi import BSBIIndex
import os
from compression import VBEPostings
from experiment import rbp, ap, dcg
import re
from gensim.models import FastText, Word2Vec
from transformers import AutoTokenizer, AutoModel

import torch
class Letor:
    def __init__(self, NUM_LATENT_TOPICS = 200):
        self.dictionary = Dictionary()
        self.NUM_LATENT_TOPICS = NUM_LATENT_TOPICS
        self.BSBI_instance = BSBIIndex(data_dir='collections (2)/collections',
                                postings_encoding=VBEPostings,
                                output_dir='index')        
        self.queries = self.open_docs_queries('qrels-folder/train_queries.txt')
        self.documents = self.open_docs_queries('qrels-folder/train_docs.txt')
        self.dataset = []
        self.group_qid_count = []
        self.bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]
        self.model = LsiModel(self.bow_corpus, num_topics = self.NUM_LATENT_TOPICS)


    def load_qrels(self, qrel_file = "qrels-folder/train_qrels.txt"):
        qrels = defaultdict(lambda: defaultdict(lambda: 0)) 
        with open(qrel_file) as file:
            for line in tqdm(file):
                q_id, doc_id, rel = line.strip().split()
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in qrels:
                        qrels[q_id] = []
                    qrels[q_id].append((doc_id, int(rel)))
        return qrels
    
    def load_test_qrels(self, qrel_file = "qrels-folder/test_qrels.txt"):
        qrels = defaultdict(lambda: defaultdict(lambda: 0)) 
        with open(qrel_file) as file:
            for line in file:
                parts = line.strip().split()
                qid = parts[0]
                did = int(parts[1])
                qrels[qid][did] = 1
        return qrels

    
    def open_docs_queries(self, path):
        ret = {}
        with open(path, encoding='utf-8') as file:
            for line in tqdm(file):
                content = line.strip().split()
                text = ' '.join(content[1:])
                tokens = re.findall(r'\b\w+\b', text.lower())
                tokens_without_stop_words = [self.BSBI_instance.pre_processing_text(token) for token in tokens]
                ret[content[0]] = tokens_without_stop_words
        return ret

    def query_dataset(self, NUM_NEGATIVES = 1):
        q_docs_rel = self.load_qrels()
        for q_id in q_docs_rel:
            docs_rels = q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries[q_id], self.documents[doc_id], rel))
            # tambahkan satu negative (random sampling saja dari documents)
            self.dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))
        return q_docs_rel
        

    
    
    def vector_rep(self, text):
        rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS
    
    def features(self, query, doc):
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]
    
    def x_y(self):
        X = []
        Y = []
        self.qrels = self.query_dataset()
        for (query, doc, rel) in self.dataset:
            X.append(self.features(query, doc))
            Y.append(rel)

        # ubah X dan Y ke format numpy array
        X = np.array(X)
        Y = np.array(Y) 
        return (X,Y)  

    def get_all_docs(self, doc_names):
        ret = []
        for loc in doc_names:
            doc_id = loc.split('\\')[-1].split('.')[0]
            with open(loc, encoding='utf-8') as file:
                text = file.read()
                ret.append((doc_id, text))
        return ret
    
    def retrieve_letor(self, model, docs, query_tokens, k=100):
        X_unseen = []
        for doc_id, doc in docs:
            X_unseen.append(self.features(query_tokens, doc.split()))
        X_unseen = np.array(X_unseen)
        scores = model.predict(X_unseen)
        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)
        top_k_results = sorted_did_scores[:k]
        return top_k_results
    
    def eval_retrieval(self, model, query_file="qrels-folder/test_queries.txt"):
        qrels = self.load_test_qrels()
        with open(query_file) as file:
            rbp_scores_tfidf = []
            dcg_scores_tfidf = []
            ap_scores_tfidf = []

            rbp_scores_bm25 = []
            dcg_scores_bm25 = []
            ap_scores_bm25 = []

            for qline in tqdm(file):
                parts = qline.strip().split()
                qid = parts[0]
                query = " ".join(parts[1:])
                tokens = re.findall(r'\b\w+\b', query.lower())
                query_tokens = [self.BSBI_instance.pre_processing_text(token) for token in tokens]
                doc_names_tfidf = [doc for (score, doc) in self.BSBI_instance.retrieve_tfidf(query, k=100)]
                docs_tfidf = self.get_all_docs(doc_names_tfidf)
                doc_names_bm25 = [doc for (score, doc) in self.BSBI_instance.retrieve_bm25(query, k=100, k1=0.2, b=0.9)]
                docs_bm25 = self.get_all_docs(doc_names_bm25)
                letor_docs_tfidf = [doc for (doc, score) in self.retrieve_letor(model, docs_tfidf, query_tokens)]
                letor_docs_bm25 = [doc for (doc, score) in self.retrieve_letor(model, docs_bm25, query_tokens)]
                """
                Evaluasi TF-IDF
                """
                ranking_tfidf = []
                for doc in letor_docs_tfidf:
                    did = int(doc)
                    # Alternatif lain:
                    # 1. did = int(doc.split("\\")[-1].split(".")[0])
                    # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                    # 3. disesuaikan dengan path Anda
                    if (did in qrels[qid]):
                        ranking_tfidf.append(1)
                    else:
                        ranking_tfidf.append(0)
                rbp_scores_tfidf.append(rbp(ranking_tfidf))
                dcg_scores_tfidf.append(dcg(ranking_tfidf))
                ap_scores_tfidf.append(ap(ranking_tfidf))

                ranking_bm25 = []
                for doc in letor_docs_bm25:
                    did = int(doc)
                    # Alternatif lain:
                    # 1. did = int(doc.split("\\")[-1].split(".")[0])
                    # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                    # 3. disesuaikan dengan path Anda
                    if (did in qrels[qid]):
                        ranking_bm25.append(1)
                    else:
                        ranking_bm25.append(0)
                rbp_scores_bm25.append(rbp(ranking_bm25))
                dcg_scores_bm25.append(dcg(ranking_bm25))
                ap_scores_bm25.append(ap(ranking_bm25))
        print("Hasil evaluasi TF-IDF terhadap 150 queries setelah letor")
        print("RBP score =", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf))
        print("DCG score =", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf))
        print("AP score  =", sum(ap_scores_tfidf) / len(ap_scores_tfidf))

        print("Hasil evaluasi BM25 terhadap 150 queries setelah letor")
        print("RBP score =", sum(rbp_scores_bm25) / len(rbp_scores_bm25))
        print("DCG score =", sum(dcg_scores_bm25) / len(dcg_scores_bm25))
        print("AP score  =", sum(ap_scores_bm25) / len(ap_scores_bm25))

if __name__ == '__main__':
    letor = Letor()

    ranker = lgb.LGBMRanker(
                    objective="lambdarank",
                    boosting_type = "gbdt",
                    importance_type = "gain",
                    metric = "ndcg")

    X,Y = letor.x_y()
    ranker.fit(X, Y, group = letor.group_qid_count)
    letor.eval_retrieval(ranker)
    