from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import docx
import textstat
import nltk
from nltk.corpus import stopwords
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            return "\n".join([page.extract_text() or '' for page in reader.pages])
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

def keyword_analysis(text):
    words = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words_filtered = [w for w in words if w.isalnum() and w not in stop_words]
    keywords = Counter(words_filtered).most_common(10)
    return f"Top keywords: {keywords}"

def check_passive_voice(text):
    # Basic heuristic (not perfect)
    passive_phrases = ["was", "were", "is being", "has been", "have been", "will be", "being"]
    sentences = nltk.sent_tokenize(text)
    passive_count = sum(1 for s in sentences if any(p in s for p in passive_phrases))
    return f"Passive voice sentences: {passive_count} of {len(sentences)}"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['resume']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    text = extract_text(filepath)
    if not text.strip():
        return render_template('index.html', feedback="Could not extract any text from the file.")

    readability = textstat.flesch_reading_ease(text)
    keywords = keyword_analysis(text)
    passive = check_passive_voice(text)

    feedback = f"Readability score (Flesch): {readability:.2f}\n\n{keywords}\n\n{passive}"
    return render_template('index.html', feedback=feedback)

if __name__ == '__main__':
    app.run(debug=True)
