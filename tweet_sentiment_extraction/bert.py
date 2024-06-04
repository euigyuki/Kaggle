import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3, ignore_mismatched_sizes=True)

# Function to perform sentiment analysis
def analyze_sentiment(sentences):
    all_predictions = []
    batch_size = 8  # Process in smaller batches to avoid memory issues

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(predictions.tolist())
    
    return all_predictions

# Read the CSV file
df = pd.read_csv("train.csv")

# Extract the text column for sentiment analysis
sentences = df['text'].astype(str).tolist()

# Perform sentiment analysis
predictions = analyze_sentiment(sentences)

# Map predictions to sentiment labels
labels = ["negative", "neutral", "positive"]
sentiment_results = [labels[prediction] for prediction in predictions]

# Add the sentiment results to the DataFrame
df['predicted_sentiment'] = sentiment_results

# Print results
for idx, row in df.iterrows():
    print(f"TextID: {row['textID']}\nText: {row['text']}\nSelected Text: {row['selected_text']}\nOriginal Sentiment: {row['sentiment']}\nPredicted Sentiment: {row['predicted_sentiment']}\n")

# Optionally, save the DataFrame with the new column to a new CSV file
df.to_csv("predicted_sentiments.csv", index=False)
