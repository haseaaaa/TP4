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
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Fungsi untuk mengambil hasil dari model BERT
def get_bert_scores(query, documents, model, tokenizer):
    inputs = tokenizer(query, documents, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = softmax(logits, dim=1)
    return probabilities[:, 1].tolist()

# Menginisialisasi model dan tokenizer BERT
bert_model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")

# ...
def load_qrels(qrel_file="qrels-folder/test_qrels.txt"):
    """ 
        memuat query relevance judgment (qrels) 
        dalam format dictionary of dictionary qrels[query id][document id],
        dimana hanya dokumen yang relevan (nilai 1) yang disimpan,
        sementara dokumen yang tidak relevan (nilai 0) tidak perlu disimpan,
        misal {"Q1": {500:1, 502:1}, "Q2": {150:1}}
    """
    with open(qrel_file) as file:
        content = file.readlines()

    qrels_sparse = {}

    for line in content:
        parts = line.strip().split()
        qid = parts[0]
        did = int(parts[1])
        if not (qid in qrels_sparse):
            qrels_sparse[qid] = {}
        if not (did in qrels_sparse[qid]):
            qrels_sparse[qid][did] = 0
        qrels_sparse[qid][did] = 1
    return qrels_sparse

def get_all_docs(doc_names):
    ret = []
    for loc in doc_names:
        doc_id = loc.split('\\')[-1].split('.')[0]
        with open(loc, encoding='utf-8') as file:
            text = file.read()
            ret.append((doc_id, text))
    return ret
def eval_retrieval_with_bert(qrels, query_file="qrels-folder/test_queries.txt", k=100):
    # ...
    BSBI_instance = BSBIIndex(data_dir='collections (2)/collections',
                            postings_encoding=VBEPostings,
                            output_dir='index')

    with open(query_file) as file:
        rbp_scores_combined = []
        dcg_scores_combined = []
        ap_scores_combined = []

        for qline in tqdm(file):
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            """
            Evaluasi BM25
            """
            ranking_bm25 = []
            # nilai k1 dan b dapat diganti-ganti
            for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k, k1=0.2, b=0.9):
                ranking_bm25.append((score, doc))

            # Ambil teks dari dokumen untuk digunakan oleh model BERT
            doc_text = get_all_docs([doc for (score,doc) in ranking_bm25])
            document_texts = [text for doc,text in(doc_text)]

            # Gunakan model BERT untuk mendapatkan skor reranking
            bert_scores = get_bert_scores(query, document_texts, bert_model, tokenizer)

            # Gabungkan skor BM25 dan BERT
            combined_scores = [bm25_score + bert_score for bm25_score, bert_score in zip([score for score, _ in ranking_bm25], bert_scores)]

            # Urutkan dokumen berdasarkan skor gabungan
            reranked_documents = [doc for _, doc in sorted(zip(bert_scores, [doc for _, doc in ranking_bm25]), reverse=True)]

            # Evaluasi seperti sebelumnya
            ranking_combined = []
            for doc in reranked_documents:
                did = int(os.path.splitext(os.path.basename(doc))[0])
                if (did in qrels[qid]):
                    ranking_combined.append(1)
                else:
                    ranking_combined.append(0)

            rbp_scores_combined.append(rbp(ranking_combined))
            dcg_scores_combined.append(dcg(ranking_combined))
            ap_scores_combined.append(ap(ranking_combined))

    # ...

    print("Hasil evaluasi BM25 + BERT terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_combined) / len(rbp_scores_combined))
    print("DCG score =", sum(dcg_scores_combined) / len(dcg_scores_combined))
    print("AP score  =", sum(ap_scores_combined) / len(ap_scores_combined))

# ...

if __name__ == '__main__':
    qrels = load_qrels()

    eval_retrieval_with_bert(qrels)

