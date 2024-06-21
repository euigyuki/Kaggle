import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import re
from sklearn.metrics import accuracy_score

# Training parameters
num_epochs = 1000
embedding_dim = 50
hidden_dim = 128
learning_rate = 0.001
batch_size = 32
train_set_size = 10
max_length = 128  # Maximum length for truncation

# Tokenizer and labels
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
SENTIMENT_LABELS = ['positive', 'neutral', 'negative']
SENTIMENT_TO_IDX = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}

class TweetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet, sentiment = self.data[idx]
        tweet = remove_urls(tweet)
        encoded_tweet = tokenizer(
            tweet,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        sentiment_idx = SENTIMENT_TO_IDX[sentiment]
        return encoded_tweet['input_ids'].squeeze(), encoded_tweet['attention_mask'].squeeze(), sentiment_idx

class SentimentNN(nn.Module):
    def __init__(self, bert_model, hidden_dim, output_dim):
        super(SentimentNN, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.pooler_output)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def load_and_preprocess_data(filepath, text_column):
    data = pd.read_csv(filepath)[[text_column, 'sentiment']]
    data[text_column] = data[text_column].fillna('')
    data = data.values.tolist()[:train_set_size]
    return data

def collate_fn(batch):
    input_ids, attention_masks, sentiments = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    sentiments = torch.tensor(sentiments)
    return input_ids, attention_masks, sentiments

def train_model(train_loader, model, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    for epoch in range(num_epochs):
        for input_ids, attention_masks, sentiments in train_loader:
            input_ids, attention_masks, sentiments = input_ids.to(device), attention_masks.to(device), sentiments.to(device)
            output = model(input_ids, attention_masks)
            loss = criterion(output, sentiments)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

def evaluate_model(test_loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for input_ids, attention_masks, sentiments in test_loader:
            input_ids, attention_masks, sentiments = input_ids.to(device), attention_masks.to(device), sentiments.to(device)
            output = model(input_ids, attention_masks)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().tolist())
            true_labels.extend(sentiments.cpu().tolist())
    accuracy = accuracy_score(true_labels, predictions)
    print("Test accuracy:", accuracy)
    return predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_data = load_and_preprocess_data('train.csv', 'selected_text')
    train_dataset = TweetDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    output_dim = len(SENTIMENT_LABELS)
    model = SentimentNN(bert_model, hidden_dim, output_dim).to(device)
 
    train_model(train_loader, model, num_epochs)

    test_data = load_and_preprocess_data('test.csv', 'text')
    test_dataset = TweetDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    evaluate_model(test_loader, model)

if __name__ == '__main__':
    main()
