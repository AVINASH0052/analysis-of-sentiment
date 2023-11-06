from flask import Flask, render_template, request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random

app = Flask(__name__)

nltk.download("vader_lexicon")
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = []
    statement=[]
    if request.method == "POST":
      user_input = request.form["user_input"]
      user=list(user_input)
      statement=user
      processed_input = preprocess_data(user)
      sentiment = analyze_sentiment(processed_input)

    return render_template("index.html", sentiment=sentiment,statement=statement)

def preprocess_data(data):
  stop_words = set(stopwords.words("english"))
  lemmatizer = WordNetLemmatizer()
  preprocessed_data = []

  for text in data:
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    preprocessed_data.append(" ".join(words))

  return preprocessed_data

def analyze_sentiment(data):
  sentiments = []
  sia = SentimentIntensityAnalyzer()
  
  for text in data:
    sentiment = sia.polarity_scores(text)
    if sentiment["compound"] >= 0.05:
        sentiments.append("positive")
    elif sentiment["compound"] <= -0.05:
        sentiments.append("negative")
    else:
        sentiments.append("neutral")

  return sentiments

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
