import os
import pickle
import contextlib
import heapq
import math
import re
from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, merge_and_sort_posts_and_tfs
from compression import VBEPostings
from tqdm import tqdm
from collections import defaultdict
from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from operator import itemgetter


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.first = True
        self.avg_doc_length = 0
        self.stemmer = MPStemmer(check_nonstandard=False)
        self.remover = StopWordRemoverFactory().create_stop_word_remover()
        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def pre_processing_text(self, content):
        """
        Melakukan preprocessing pada text, yakni stemming dan removing stopwords
        """
        # https://github.com/ariaghora/mpstemmer/tree/master/mpstemmer

        stemmed = self.stemmer.stem(content)
        return self.remover.remove(stemmed)

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk stemming bahasa Indonesia, seperti
        MpStemmer: https://github.com/ariaghora/mpstemmer 
        Jangan gunakan PySastrawi untuk stemming karena kode yang tidak efisien dan lambat.

        JANGAN LUPA BUANG STOPWORDS! Kalian dapat menggunakan PySastrawi 
        untuk menghapus stopword atau menggunakan sumber lain seperti:
        - Satya (https://github.com/datascienceid/stopwords-bahasa-indonesia)
        - Tala (https://github.com/masdevid/ID-Stopwords)

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        # TODO
        td_pairs = []

        full_path = os.path.join(self.data_dir,block_path)
        # Loop melalui file-file dalam direktori blok
        for filename in os.listdir(full_path):
            if filename.endswith(".txt"):
                with open(os.path.join(full_path, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    
                    tokens = re.findall(r'\b\w+\b', text.lower())
                    
                    tokens_without_stop_words = [self.pre_processing_text(token) for token in tokens]
                    
                    term_ids = [self.term_id_map[term] for term in tokens_without_stop_words]
                    
                    doc_id = self.doc_id_map[os.path.join(full_path, filename)]
                    
                    td_pairs.extend([(term_id, doc_id) for term_id in term_ids])

        return td_pairs

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # TODO
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = {}
            if doc_id in term_dict[term_id]:
                term_dict[term_id][doc_id] = term_dict[term_id][doc_id] + 1
            else:
                term_dict[term_id][doc_id] = 1
        for term_id in sorted(term_dict.keys()):
            postings_list = sorted(list(term_dict[term_id]))
            tf_list = [term_dict[term_id][doc_id] for doc_id in postings_list]
            index.append(term_id, postings_list, tf_list)


    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        scores = defaultdict(float)
        self.load()
        tokens = re.findall(r'\b\w+\b', query.lower())
        query_tokens = [self.pre_processing_text(token) for token in tokens]

        idf_values = {}
        
        terms = [self.term_id_map[token] for token in query_tokens]
        
        with InvertedIndexReader('main_index', self.postings_encoding, directory=self.output_dir) as reader:
            for term in terms:
                if term in reader.postings_dict:
                    N = len(reader.doc_length)
                    pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = reader.postings_dict[term]
                    idf = math.log10(N / number_of_postings)
                    idf_values[term] = idf
            reader.reset()
        
        with InvertedIndexReader('main_index', self.postings_encoding, directory=self.output_dir) as reader:
            for term in terms:
                if term in reader.postings_dict:
                    pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = reader.postings_dict[term]
                    postings_list, tf_list = reader.get_postings_list(term)
                    for i in range(len(postings_list)):
                        tf_weight = 1 + math.log10(tf_list[i])
                        idf_weight = idf_values[term]

                        scores[postings_list[i]] += tf_weight * idf_weight
            reader.reset()
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k_results = sorted_scores[:k]
        
        doc_names = [self.doc_id_map[doc_id] for doc_id, _ in top_k_results]

        
        return [(score, doc_name) for (doc_id, score), doc_name in zip(top_k_results, doc_names)]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan framework TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        """
        # TODO

        scores = defaultdict(float)
        self.load()
        tokens = re.findall(r'\b\w+\b', query.lower())
        query_tokens = [self.pre_processing_text(token) for token in tokens]

        idf_values = {}
        
        terms = [self.term_id_map[token] for token in query_tokens]
        
        with InvertedIndexReader('main_index', self.postings_encoding, directory=self.output_dir) as reader:
            for term in terms:   
                if self.first:
                    self.avg_doc_length = sum(reader.doc_length.values()) / len(reader.doc_length)
                    self.first = False
                if term in reader.postings_dict:
                    N = len(reader.doc_length)
                    pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = reader.postings_dict[term]
                    idf = math.log10(N / number_of_postings)
                    idf_values[term] = idf
            reader.reset()

        
        with InvertedIndexReader('main_index', self.postings_encoding, directory=self.output_dir) as reader:
            for term in terms:
                if term in reader.postings_dict:
                    pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = reader.postings_dict[term]
                    postings_list, tf_list = reader.get_postings_list(term)
                    for i in range(len(postings_list)):
                        doc_id = postings_list[i]
                        tf = tf_list[i]
                        doc_length = reader.doc_length[doc_id]
                        idf_weight = idf_values[term]
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
                        scores[doc_id] += idf_weight * (numerator / denominator)
            reader.reset()

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k_results = sorted_scores[:k]

        doc_names = [self.doc_id_map[doc_id] for doc_id, _ in top_k_results]

        return [(score, doc_name) for (doc_id, score), doc_name in zip(top_k_results, doc_names)]

    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='collections (2)/collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.do_indexing()  # memulai indexing!
