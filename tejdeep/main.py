import threading
import http.server
import socketserver
from flask_cors import CORS
from flask import Flask, request, jsonify
import os
import uuid
import numpy as np
import torch
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine
import soundfile as sf

app = Flask(__name__)
UPLOAD_FOLDER = 'tejdeep\\voice_db'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def run_http_server():
    """Runs a simple HTTP server in a separate thread."""
    PORT = 8080
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving static files at http://localhost:{PORT}")
        httpd.serve_forever()
        
        
# Load pretrained speaker recognition model
spk_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def compute_embedding(filepath):
    signal, fs = sf.read(filepath)
    
    # Ensure it's a mono channel
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Convert to float32 to ensure model compatibility
    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # Shape: [1, time]
    
    with torch.no_grad():
        embedding = spk_model.encode_batch(signal)
    
    return embedding.squeeze().numpy()



# Simple in-memory storage for demo
embeddings_db = {}

@app.route("/enroll", methods=["POST"])
def enroll_voice():
    """Add a new voice sample to the DB"""
    file = request.files["file"]
    name = request.form["name"]
    filename = os.path.join(UPLOAD_FOLDER, f"{name}_{uuid.uuid4().hex}.wav")
    file.save(filename)
    emb = compute_embedding(filename)
    embeddings_db[filename] = emb
    return jsonify({"message": "Voice enrolled", "file": filename})

@app.route("/match", methods=["POST"])
def match_voice():
    """Compare a new voice sample to all enrolled voices"""
    file = request.files["file"]
    query_path = os.path.join(UPLOAD_FOLDER, f"query_{uuid.uuid4().hex}.wav")
    file.save(query_path)
    query_emb = compute_embedding(query_path)

    best_score = -1
    best_match = None

    for ref_file, ref_emb in embeddings_db.items():
        similarity = 1 - cosine(query_emb, ref_emb)  # Cosine similarity
        if similarity > best_score:
            best_score = similarity
            best_match = ref_file

    return jsonify({
        "match_file": best_match,
        "match_percentage": round(best_score * 100, 2)
    })

if __name__ == "__main__":
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    CORS(app)
    app.run(debug=True)
    
