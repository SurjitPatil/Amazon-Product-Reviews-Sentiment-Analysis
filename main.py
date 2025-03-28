import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Task 1: Decoding Amazon Reviews ---
df = pd.read_csv(r"C:\Users\SURJIT PATIL\Downloads\Reviews.csv")

# --- Task 2: Preparing for Sentiment Analysis ---
df.dropna(inplace=True)

# --- Task 3: Visualizing Sentiments ---
ratings = df['Score'].value_counts().sort_values(ascending=False)
numbers = ratings.index
quantity = ratings.values

custom_colors = ["skyblue", "yellowgreen", 'tomato', "blue", "red"]
plt.figure(figsize=(10, 8))
plt.pie(quantity, labels=numbers, colors=custom_colors)
central_circle = plt.Circle((0, 0), 0.5, color='white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size=12)
plt.title("Distribution of Amazon Product Ratings", fontsize=20)
plt.show()

# --- Task 4: Exploring Amazon Product Reviews ---
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()

df[['Positive', 'Negative', 'Neutral']] = df['Text'].apply(
    lambda x: pd.Series(sentiments.polarity_scores(str(x)))
).loc[:, ['pos', 'neg', 'neu']]

# --- Task 5: Unmasking Amazon's Dominant Emotion ---
x, y, z = df['Positive'].sum(), df['Negative'].sum(), df['Neutral'].sum()

sentiment = 'Positive' if x > y and x > z else 'Negative' if y > x and y > z else 'Neutral'
