import pandas as pd
import nltk
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# Load the CSV file into a DataFrame
file_path = "7817_1.csv"
df = pd.read_csv(file_path)

# Define preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    elif not isinstance(text, str):
        return str(text)

    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_review'] = df['reviews.text'].apply(preprocess_text)

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define function to get sentiment scores
def get_sentiment(cleaned_review):
    return sid.polarity_scores(cleaned_review)

# Apply sentiment analysis
df['sentiment'] = df['cleaned_review'].apply(get_sentiment)

# Flask app setup
app = Flask(__name__, static_url_path='', static_folder='templates')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/reviews', methods=['GET'])
def get_reviews():
    review_data = []
    for index, row in df.iterrows():
        review_data.append({
            'text': row['reviews.text'],
            'sentiment': get_sentiment(row['cleaned_review']),
            'summary': summarize_review_nltk(row['cleaned_review'])
        })

    return jsonify(review_data)

# Define function to summarize a review using NLTK
def summarize_review_nltk(review_text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(review_text)
    
    # Select a fraction of most informative sentences
    num_sentences = int(len(sentences) * 0.2)  # 20% of original text
    if num_sentences < 1:
        num_sentences = 1  # Ensure at least one sentence is selected
    
    # Score each sentence based on its length
    ranked_sentences = [(sentence, len(sentence)) for sentence in sentences]
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[1], reverse=True)
    
    # Select top sentences as summary
    top_sentences = ranked_sentences[:num_sentences]
    summary = ' '.join(sentence for sentence, _ in top_sentences)
    
    return summary

if __name__ == '__main__':
    app.run(debug=True)
