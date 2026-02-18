import sys
import os
from pathlib import Path
import numpy as np

# Add Resemblyzer to path so it can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'Resemblyzer'))

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from resemblyzer import VoiceEncoder, preprocess_wav
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

encoder = VoiceEncoder()

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def compute_similarity(source_path, target_paths):
    # Use resemblyzer's preprocessing (Normalizes volume, trims silence, resamples)
    source_wav = preprocess_wav(source_path)
    source_embed = encoder.embed_utterance(source_wav)

    similarities = []
    for path in target_paths:
        try:
            wav = preprocess_wav(path)
            embed = encoder.embed_utterance(wav)
            similarity = np.dot(source_embed, embed)
            # Ensure score is a python float
            score = round(float(similarity) * 100, 2)
            similarities.append((os.path.basename(path), score))
        except Exception as e:
            similarities.append((os.path.basename(path), f"Error: {e}"))

    # Separate successes and failures
    successes = [s for s in similarities if isinstance(s[1], (int, float))]
    failures = [s for s in similarities if not isinstance(s[1], (int, float))]

    sorted_successes = sorted(successes, key=lambda x: x[1], reverse=True)
    final_results = sorted_successes + failures

    return final_results, sorted_successes[0] if sorted_successes else None

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/checker', methods=['GET', 'POST'])
def voice_checker():
    if request.method == 'POST':
        source_file = request.files['source']
        target_files = request.files.getlist('targets')

        if not source_file or not target_files:
            return render_template('voice_checker.html', error="Please upload both source and target files.")

        source_filename = secure_filename(source_file.filename)
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], source_filename)
        source_file.save(source_path)

        target_paths = []
        target_filenames = []
        for file in target_files:
            target_filename = secure_filename(file.filename)
            target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_filename)
            file.save(target_path)
            target_paths.append(target_path)
            target_filenames.append(target_filename)

        results, most_similar = compute_similarity(source_path, target_paths)

        # Include source file in results for comparison
        source_result = (source_filename, 100.0)  # Source is 100% similar to itself
        all_results = [source_result] + results

        return render_template('voice_checker.html', results=all_results, most_similar=most_similar, source_filename=source_filename)

    return render_template('voice_checker.html')

if __name__ == '__main__':
    app.run(debug=True)
