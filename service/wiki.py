#!pip3 install git+https://github.com/aatimofeev/spacy_russian_tokenizer.git
from spacy.lang.ru import Russian
from tqdm import tqdm_notebook as tqdm
import gensim
from gensim.test.utils import datapath
from collections import defaultdict
import pickle
import json
import numpy as np
import math


wv_from_bin = gensim.models.Word2Vec.load("data/ru.bin")

CRITIAL_WORDS = ['лучшее', 'самое', 'важный']
CRITIAL_WORDS_vw = [wv_from_bin[x] for x in CRITIAL_WORDS]

docs_by_tokens = None
ps = None
count_total = None
avgdl = None

def prepare(CHUNK):
    global docs_by_tokens, ps, count_total, avgdl
    with open('data/wiki_doc_by_tokens{}.txt'.format(CHUNK)) as f:
        docs_by_tokens = json.load(f)

    with open('data/wiki_texts{}.txt'.format(CHUNK)) as f:
        ps = json.load(f)

    from collections import defaultdict

    count_total = sum([len(docs_by_tokens[t]) for t in docs_by_tokens.keys()])
    print(count_total)

    avgdl = sum([len(p) for p in ps]) / len(ps)
    print('avgdl', avgdl)

nlp = Russian()


def is_doc_ok(query, doc, rarest_token):
    dc_tkns = nlp(doc)
    if rarest_token not in [token.text for token in dc_tkns]:
        print('No rearest')
        return False
    if len(doc) > len(query) * 10 or len(doc) < 7:
        print('Size diff too big')
        return False
    return True
    #wvs = []
    #for x in dc_tkns:
    #    try:
    #        wvs += [wv_from_bin[x]]
    #    except:
    #        pass
   # 
    # crit words
    #if not any([np.sum((c - )**2) for c in CRITIAL_WORDS_vw])

def get_score(doc, cnt_in_doc, q_docs):
    idf = math.log((len(ps) - q_docs + 0.5) / (q_docs + 0.5))
    return idf * cnt_in_doc * (3.5) / (cnt_in_doc + 2*(0.25 + .75 * len(ps)/avgdl)) # bm25

def get_best_doc(query = 'ступени в ракете'):
    query = query.lower()
    tokens =[token.text for token in nlp(query)]
    rarest_tokens = sorted(tokens, key=lambda k: len(docs_by_tokens.get(k, [])))
    rarest_token = None
    for t in rarest_tokens:
        if t in docs_by_tokens:
            rarest_token = t
            break

    print(tokens, rarest_token)

    if rarest_token is None:
        return None

    best_docs = defaultdict(int)

    for i in range(len(tokens)):
        print(i)
        if i > 0:
            bigram = ' '.join([tokens[i - 1], tokens[i]])
            docs = docs_by_tokens.get(bigram, [])
            for d, cnt in docs_by_tokens.get(bigram, {}).items():
                d = int(d)
                best_docs[d] += get_score(ps[d], cnt, len(docs)) * is_doc_ok(query, ps[d], rarest_token)                
        docs = docs_by_tokens.get(tokens[i], {}) 
        #print(docs)
        print(i)
        print(len(docs))
        for d, cnt in docs.items():
            d = int(d)
            best_docs[d] += get_score(ps[d], cnt, len(docs)) #* is_doc_ok(query, ps[d], rarest_token)

    best_docs2 = {}

    for d in docs_by_tokens.get(rarest_token, {}):
        d = int(d)
        best_docs2[d] = best_docs[d]


    if len(best_docs2) == 0:
        return None
    best_doc = sorted(best_docs2.items(), key=lambda k: k[1])[-1]

    return ps[best_doc[0]], is_doc_ok(query, ps[best_doc[0]], rarest_token), best_doc[1]
