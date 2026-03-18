from flask import Flask, render_template, request
from search_engine import PengugleSearchEngine
import time

app = Flask(__name__)

# Cargar el motor de búsqueda al arrancar el servidor
engine = PengugleSearchEngine('corpus.json')

@app.route('/', methods=['GET'])
def index():
    query = request.args.get('q', '')
    results = []
    stats = {
        'total_docs': engine.total_docs,
        'vocab_size': len(engine.vocab),
        'search_time': 0
    }
    
    if query:
        start_time = time.time()
        results = engine.search_bm25(query)
        stats['search_time'] = round(time.time() - start_time, 4)
        
    return render_template('index.html', query=query, results=results, stats=stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)