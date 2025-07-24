
from flask_cors import CORS
from flask import Flask, request, jsonify ,render_template
import os
import uuid
import numpy as np
import torch
from speechbrain.inference import SpeakerRecognition
from scipy.spatial.distance import cosine
import soundfile as sf

app = Flask(__name__)
UPLOAD_FOLDER = 'tejdeep/voice_db'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pretrained speaker recognition model
spk_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


def compute_embedding(filepath):
    signal, fs = sf.read(filepath)

    # Ensure it's a mono channel
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Convert to float32 to ensure model compatibility
    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(
        0)  # Shape: [1, time]

    with torch.no_grad():
        embedding = spk_model.encode_batch(signal)

    return embedding.squeeze().numpy()


# Simple in-memory storage for demo
embeddings_db = {}

@app.route('/')
def index():
    return render_template('uploadwav.html')


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
    try:
        file = request.files["file"]
        query_path = os.path.join(UPLOAD_FOLDER, f"query_{uuid.uuid4().hex}.wav")
        file.save(query_path)
        query_emb = compute_embedding(query_path)

        best_score = -1
        best_match = None

        for ref_file, ref_emb in embeddings_db.items():
            similarity = 1 - cosine(query_emb, ref_emb)
            if similarity > best_score:
                best_score = similarity
                best_match = ref_file

        if best_match is None:
            return jsonify({"error": "No enrolled voices to compare against."}), 400

        return jsonify({
            "match_file": best_match,
            "match_percentage": round(float(best_score) * 100, 2)

        })

    except Exception as e:
        print("🔥 Internal error in /match:", e)
        return jsonify({"error": str(e)}), 500
    
    
    


@app.route('/api/read-file', methods=['GET'])
def read_file():
    file_path = os.path.expanduser("./caller.txt")
    result = []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            # Skip header row
            for line in lines[1:]:

                if len(columns) >= 9:
                    columns = line.strip().split()
                    caller_id = columns[7]
                    timestamp = columns[8]
                    appended_location = f"{columns[6]}/{caller_id}-{timestamp}.wav"
                entry = {
                    "Name": "Teena",  # 🔹 Add the constant Name field
                    "Phone no.": columns[2],
                    "Status": columns[5],
                    "Location": appended_location,
                    "CallerID": columns[7],
                    "Duration": columns[8]
                }
                result.append(entry)

        return jsonify({"data": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
#    http_thread = threading.Thread(target=run_http_server, daemon=True)
#    http_thread.start()
    CORS(app)
    app.run(debug=True, host="0.0.0.0")