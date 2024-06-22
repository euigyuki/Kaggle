import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
from sklearn.metrics import accuracy_score
from vocab import Vocabulary
from tweetdataset import TweetTrainDataset, TweetTestDataset
from nltk.tokenize import word_tokenize


# Training parameters
tokenizer = word_tokenize

num_epochs = 25
embedding_dim = 200
hidden_dim = 128
learning_rate = 0.002
batch_size = 32
CHECKING_TRAIN_SIZE = False
CHECKING_TEST_SIZE = False
print_statements = False


# Tokenizer and labels
DISASTER_LABELS = ['no_disaster', 'disaster']
DISASTER_TO_IDX = {label: idx for idx, label in enumerate(DISASTER_LABELS)}
paddings = []
max_length = 15
train_set_size = 2
test_set_size = 6

class AttentionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(AttentionBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context_vector)

class SentimentNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True,dropout=0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])


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

def train_collate_fn(batch):
    tweets, sentiments = zip(*batch)
    padded_tweets = pad_sequence(tweets, batch_first=True, padding_value=1)
    sentiments = torch.tensor(sentiments)
    return padded_tweets, sentiments

def test_collate_fn(batch):
    padded_tweets = pad_sequence(batch, batch_first=True, padding_value=1)
    return padded_tweets

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

def sort_dataset_by_length(dataset):
    lengths = [len(tokens) for tokens,  _ in dataset]
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    sorted_data = [dataset[i] for i in sorted_indices]
    return sorted_data

def load_and_preprocess_train_data():
    data = pd.read_csv('train.csv')[['text', 'target']]
    data['text'] = data['text'].fillna('')
    if CHECKING_TRAIN_SIZE:
        data = data.values.tolist()[:train_set_size]
    else:
        data = data.values.tolist()
    return data

def load_and_preprocess_test_data():
    data = pd.read_csv('test.csv')[['id', 'text']] 
    data['text'] = data['text'].fillna('')
    if CHECKING_TEST_SIZE:
        data = data.values.tolist()[:test_set_size]
    else:
        data = data.values.tolist()
    return data

def create_vocab(data):
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    for tweet,_ in data:
        for token in tokenizer(tweet):
            token = token.lower()
            vocab.add_word(token)        
    print("Vocabulary size:", len(vocab))
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

def evaluate_model(test_loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = []
    with torch.no_grad():
        for tweets in test_loader:
            tweets = tweets.to(device)
            output = model(tweets)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().tolist())
    return predictions

def create_submission(predictions, test_data, output_file):
    submission = pd.DataFrame({
        'id': [row[0] for row in test_data],  
        'target': [pred for pred in predictions]
    })
    submission.to_csv(output_file, index=False)
    print(f"Submission file created: {output_file}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_data = load_and_preprocess_train_data()
    test_data = load_and_preprocess_test_data()

    vocab = create_vocab(train_data)
    train_dataset = TweetTrainDataset(train_data, vocab)
    test_dataset = TweetTestDataset(test_data, vocab)

    #train_dataset = sort_dataset_by_length(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_collate_fn)

    vocab_size = len(vocab)
    output_dim = len(DISASTER_LABELS)
    #model = SentimentNN(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
    model = AttentionBiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
 
    train_model(train_loader, model, num_epochs)


    predictions = evaluate_model(test_loader, model)
    create_submission(predictions, test_data, 'final_submission.csv')

    evaluate_model(test_loader, model)
    print("Average padding percentage:", sum(paddings) / len(paddings))

if __name__ == '__main__':
    main()