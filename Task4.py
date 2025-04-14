from textblob import TextBlob
import pandas as pd

reviews = [
    "The movie was absolutely fantastic!",
    "Terrible acting and boring plot.",
    "It was okay, nothing special."
]

df = pd.DataFrame({'text': reviews})

df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['result'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative')

print("Sentiment Analysis Results:")
print(df[['text', 'result', 'sentiment']])