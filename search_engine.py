import json
import math
import re
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class PengugleSearchEngine:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.documents = []
        self.vocab = set()
        self.inverted_index = defaultdict(dict)
        self.doc_lengths = {}
        self.avgdl = 0
        self.total_docs = 0
        
        # NLP tools (English)
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # BM25 Parameters
        self.k1 = 1.5
        self.b = 0.75
        
        self.load_and_index()

    def load_and_index(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
            
        self.total_docs = len(self.documents)
        total_length = 0
        
        for doc in self.documents:
            doc_id = doc['id']
            tokens = self.preprocess(doc['text'])
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1
                self.vocab.add(token)
                
            for term, freq in term_freqs.items():
                self.inverted_index[term][doc_id] = freq
                
        if self.total_docs > 0:
            self.avgdl = total_length / self.total_docs

    def preprocess(self, text):
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return [self.stemmer.stem(t) for t in tokens if t not in self.stop_words]

    def search_bm25(self, query):
        query_tokens = self.preprocess(query)
        scores = defaultdict(float)
        
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
                
            df = len(self.inverted_index[token])
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            for doc_id, freq in self.inverted_index[token].items():
                dl = self.doc_lengths[doc_id]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                scores[doc_id] += idf * (numerator / denominator)
                
        ranked_doc_ids = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        results = []
        # Obtenemos los términos originales de la búsqueda para el resaltado
        raw_query_terms = [w for w in re.findall(r'\b[a-zA-Z]+\b', query) if w.lower() not in self.stop_words]

        for doc_id, score in ranked_doc_ids:
            doc = next((d for d in self.documents if d['id'] == doc_id), None)
            if doc:
                highlighted_text = doc['text']
                # Enhancement A: Term Highlighting
                for word in raw_query_terms:
                    pattern = re.compile(rf'\b({word})\b', re.IGNORECASE)
                    highlighted_text = pattern.sub(r'<mark>\1</mark>', highlighted_text)
                
                results.append({
                    'id': doc['id'],
                    'title': doc['title'],
                    'text': highlighted_text,
                    'score': round(score, 4),
                    'source': doc['source']
                })
                
        return results