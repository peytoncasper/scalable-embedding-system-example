from flask import Flask, request
from sentence_transformers import SentenceTransformer
import threading
import time
from queue import Queue
from flask import jsonify

app = Flask(__name__)
queue = Queue(maxsize=32)
model = SentenceTransformer('all-MiniLM-L6-v2')

def process_queue():
    while True:
        if not queue.empty():
            sentences = []
            while not queue.empty():
                sentences.append(queue.get())
            embeddings = model.encode(sentences, batch_size=len(sentences))

            print("Generated " + str(len(embeddings)) + " Embeddings")
            # Do something with the embeddings, e.g., store in a database or perform further analysis
        time.sleep(.2)

@app.route('/embed', methods=['POST'])
def embed():
    sentence = request.get_json()['blob']
    queue.put(sentence)
    return jsonify({
        "status": "success"
    })

if __name__ == '__main__':
    thread = threading.Thread(target=process_queue)
    thread.start()
    app.run()
