import pandas as pd
import torch
import torch.nn as nn
import torchtext
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import re
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize

torchtext.disable_torchtext_deprecation_warning()

# Training parameters
num_epochs = 10
embedding_dim = 50
hidden_dim = 128
learning_rate = 0.001
batch_size = 2
train_set_size = 4

# Tokenizer and labels
tokenizer = word_tokenize
SENTIMENT_LABELS = ['positive', 'neutral', 'negative']
SENTIMENT_TO_IDX = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
paddings = []
print_statements = False
max_length = 5

class TweetDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        self.pad_idx = vocab['<pad>']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet, sentiment = self.data[idx]
        tweet = remove_urls(tweet)
        tokenized_tweet = tokenizer(tweet)
        indexed_tweet = self.vocab(tokenized_tweet)
        sentiment_idx = SENTIMENT_TO_IDX[sentiment]
        indexed_tweet = truncate_sequences(indexed_tweet, max_length)
        tweet_tensor = torch.tensor(indexed_tweet, dtype=torch.long)
        return tweet_tensor, sentiment_idx

class SentimentNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def yield_tokens(data):
    for tweet, _ in data:
        yield tokenizer(tweet)

def analyze_padding(tweet_tensors, pad_value=1):
    total_tokens = 0
    total_padding = 0
    if print_statements:
        print(f"Analyzing padding...")
        print("Shape of tweet_tensors:", tweet_tensors.size())
    
    for tweet in tweet_tensors:
        if print_statements:
            print("\nTweet:", tweet)
        tweet_length = tweet.size(0)
        padding_count = (tweet == pad_value).sum().item()
        total_tokens += tweet_length
        total_padding += padding_count
        padding_percentage = (padding_count / tweet_length) * 100
        if print_statements:
            print(f"Tweet length: {tweet_length}, Padding: {padding_count} ({padding_percentage:.2f}%)")
    
    overall_padding_percentage = (total_padding / total_tokens) * 100
    if print_statements:
        print(f"\nOverall stats:")
        print(f"Total tokens: {total_tokens}")
        print(f"Total padding: {total_padding}")
        print(f"Overall padding percentage: {overall_padding_percentage:.2f}%")
    paddings.append(overall_padding_percentage)

def collate_fn(batch):
    tweets, sentiments = zip(*batch)
    padded_tweets = pad_sequence(tweets, batch_first=True, padding_value=1)
    sentiments = torch.tensor(sentiments)
    return padded_tweets, sentiments

def truncate_sequences(sequences, max_length):
    if print_statements:
        print("Truncating sequences...")
        print("Max length:", max_length)
        print("Sequences:", sequences," and length:", len(sequences))
    length = min(max_length, len(sequences))
    result = sequences[:length]
    if print_statements:
        print("Result:", result)
    return result

def load_and_preprocess_data(filepath, text_column):
    data = pd.read_csv(filepath)[[text_column, 'sentiment']]
    data[text_column] = data[text_column].fillna('')
    data = data.values.tolist()#[:train_set_size]
    return data

def create_vocab(data):
    vocab = build_vocab_from_iterator(yield_tokens(data), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def train_model(train_loader, model, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    for epoch in range(num_epochs):
        for tweets, sentiments in train_loader:
            analyze_padding(tweets)
            tweets, sentiments = tweets.to(device), sentiments.to(device)
            output = model(tweets)
            loss = criterion(output, sentiments)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# def evaluate_model(test_loader, model):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.eval()
#     predictions = []
#     true_labels = []
#     with torch.no_grad():
#         for tweets, sentiments in test_loader:
#             tweets, sentiments = tweets.to(device), sentiments.to(device)
#             output = model(tweets)
#             _, predicted = torch.max(output, 1)
#             predictions.extend(predicted.cpu().tolist())
#             true_labels.extend(sentiments.cpu().tolist())
#     accuracy = accuracy_score(true_labels, predictions)
#     print("Test accuracy:", accuracy)
#     return predictions

def evaluate_model(test_loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = []
    with torch.no_grad():
        for tweets, _ in test_loader:
            tweets = tweets.to(device)
            output = model(tweets)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().tolist())
    return predictions

def create_submission(predictions, test_data, output_file):
    submission = pd.DataFrame({
        'textID': [row[0] for row in test_data],  # Assuming the first column is textID
        'sentiment': [SENTIMENT_LABELS[pred] for pred in predictions]
    })
    submission.to_csv(output_file, index=False)
    print(f"Submission file created: {output_file}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_data = load_and_preprocess_data('train.csv', 'selected_text')
    vocab = create_vocab(train_data)
    train_dataset = TweetDataset(train_data, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # for tweets, sentiments in train_loader:
    #     print("train Tweets:", tweets)
    #     print("train Sentiments:", sentiments)

    vocab_size = len(vocab)
    output_dim = len(SENTIMENT_LABELS)
    model = SentimentNN(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
 
    train_model(train_loader, model, num_epochs)

    test_data = load_and_preprocess_data('test.csv', 'text')
    test_dataset = TweetDataset(test_data, vocab)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # for tweets, sentiments in test_loader:
    #     print("test Tweets:", tweets)
    #     print("test Sentiments:", sentiments)

    predictions = evaluate_model(test_loader, model)
    create_submission(predictions, test_data, 'sample_submission.csv')

    evaluate_model(test_loader, model)
    print("Average padding percentage:", sum(paddings) / len(paddings))

if __name__ == '__main__':
    main()
