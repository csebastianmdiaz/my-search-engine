import json #read corpus
import math #math calculation in BM25 formula 
import re #regular expression
import nltk #NLP
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class PengugleSearchEngine:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path #save path JSON
        self.documents = [] #store all the texts
        self.vocab = set() #track unique words
        self.inverted_index = defaultdict(dict)
        self.doc_lengths = {}
        self.avgdl = 0 #average length all documents
        self.total_docs = 0
        
        #NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        #BM25 parameters
        self.k1 = 1.5
        self.b = 0.75
        
        self.load_and_index()

    def load_and_index(self):
        #read JSON and load the data
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        #count how many documents
        self.total_docs = len(self.documents)
        #counter to sum all words from all docs
        total_length = 0
        
        #reviewing doc by doc
        for doc in self.documents:
            doc_id = doc['id'] #get the ID of the ecurrent doc
            #text to be cleaned
            tokens = self.preprocess(doc['text'])
            
            #save how many clean words this doc has
            self.doc_lengths[doc_id] = len(tokens)
            #add to the total words
            total_length += len(tokens)

            #dictionary to know how many times each words appear
            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1
                self.vocab.add(token)
            
            #for each word found in this document and its frequency
            for term, freq in term_freqs.items():
                #save it in the master dictionary associating the word - ID - frequency
                self.inverted_index[term][doc_id] = freq

        #calculate the average document length
        if self.total_docs > 0:
            self.avgdl = total_length / self.total_docs

    def preprocess(self, text):
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return [self.stemmer.stem(t) for t in tokens if t not in self.stop_words]

    def search_bm25(self, query):
        #clean user query
        query_tokens = self.preprocess(query)
        #dictionary to store the score each document will get
        scores = defaultdict(float)
        
        #analyze search query word by word
        for token in query_tokens:
            #word doesnt exist, move to the next one
            if token not in self.inverted_index:
                continue
            #number of docs that contain the word
            df = len(self.inverted_index[token])
            #gives higher score to rare words and lower to common ones
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            #review all docs that contain the word
            for doc_id, freq in self.inverted_index[token].items():
                dl = self.doc_lengths[doc_id] #how long the doc is
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                scores[doc_id] += idf * (numerator / denominator)
        #sort results from highest score to lowest   
        ranked_doc_ids = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        results = []

        #enhancement: A
        #extract the original word of the query
        raw_query_terms = [w for w in re.findall(r'\b[a-zA-Z]+\b', query) if w.lower() not in self.stop_words]

        #iterate through the winning documents to prepare their text
        for doc_id, score in ranked_doc_ids:
            #look for the complete original text
            doc = next((d for d in self.documents if d['id'] == doc_id), None)
            if doc:
                highlighted_text = doc['text']#make a copy of the original text

                #for each original word the user searched for
                for word in raw_query_terms:
                    #create an exact pattern that ignores uppercase and lowercase
                    pattern = re.compile(rf'\b({word})\b', re.IGNORECASE)
                    #replace that word by wrapping it in the HTML tag to paint it yellow
                    highlighted_text = pattern.sub(r'<mark>\1</mark>', highlighted_text)
                
                #all the final results
                results.append({
                    'id': doc['id'],
                    'title': doc['title'],
                    'text': highlighted_text,
                    'score': round(score, 4),
                    'source': doc['source']
                })
                
        return results