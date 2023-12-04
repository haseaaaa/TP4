from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut

BSBI_instance = BSBIIndex(data_dir='collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')
queries = ["Jumlah uang terbatas yang telah ditentukan sebelumnya bahwa seseorang harus membayar dari tabungan mereka sendiri",
           "Terletak sangat dekat dengan khatulistiwa", "sirap batu tulis"]
'''
BSBI_instance = BSBIIndex(data_dir='col_test',
                          postings_encoding=VBEPostings,
                          output_dir='index_test')

queries = ["memory operating system"]
'''
for query in queries:
    print("Query  : ", query)
    print("Results:")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query):
        print(f"{doc:30} {score:>.3f}")
    print()
