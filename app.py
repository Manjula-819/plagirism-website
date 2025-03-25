from flask import Flask, render_template, request
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def vectorize(texts):
    return TfidfVectorizer().fit_transform(texts).toarray()

def similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])[0][1]

def check_plagiarism(files):
    texts = [open(os.path.join(UPLOAD_FOLDER, f), 'r', encoding='utf-8').read() for f in files]
    vectors = vectorize(texts)
    s_vectors = list(zip(files, vectors))
    results = []
    for i, (file_a, vec_a) in enumerate(s_vectors):
        for file_b, vec_b in s_vectors[i+1:]:
            sim_score = similarity(vec_a, vec_b)
            results.append((file_a, file_b, sim_score))
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        files = request.files.getlist('files')
        if len(files) < 2:
            return render_template('index.html', error="Upload at least 2 files.")
        uploaded_files = []
        for file in files:
            if file and file.filename.endswith('.txt'):
                filename = file.filename
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                uploaded_files.append(filename)
        if uploaded_files:
            results = check_plagiarism(uploaded_files)
            return render_template('index.html', results=results)
        return render_template('index.html', error="No valid .txt files uploaded.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
