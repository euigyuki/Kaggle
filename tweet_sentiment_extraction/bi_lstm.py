import pandas as pd
import torch
import torchtext
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import torch.nn.utils.rnn as rnn_utils

torchtext.disable_torchtext_deprecation_warning()

# Training parameters
num_epochs = 1
embedding_dim = 100
hidden_dim = 128
learning_rate = 0.001
batch_size = 32
train_set_size = 3

# Tokenizer and labels
tokenizer = get_tokenizer('basic_english')
SENTIMENT_LABELS = ['positive', 'neutral', 'negative']
SENTIMENT_TO_IDX = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}


class TweetDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        self.pad_idx = vocab['<pad>']
        self.max_len = max(len(self.vocab(tokenizer(tweet))) for tweet, _ in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet, sentiment = self.data[idx]
        tokenized_tweet = tokenizer(tweet)
        indexed_tweet = self.vocab(tokenized_tweet)
        tweet_ids = indexed_tweet + [self.pad_idx] * (self.max_len - len(indexed_tweet))
        sentiment_idx = SENTIMENT_TO_IDX[sentiment]
        tweet_tensor = torch.tensor(tweet_ids, dtype=torch.long)
        return tweet_tensor, sentiment_idx

class SentimentNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, max_len):
        super(SentimentNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * max_len, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.view(embedded.size(0), -1)  # Flatten the embeddings
        out = self.fc1(embedded)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# class SentimentLSTM(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
#         super(SentimentLSTM, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)

#     def forward(self, text, text_lengths):
#         embeddings = self.embedding(text)
#         print("text lengths ",text_lengths)
#         packed_input = rnn_utils.pack_padded_sequence(embeddings, text_lengths, batch_first=True, enforce_sorted=False)
#         print(packed_input.batch_sizes)
#         outputs, (hidden, _) = self.lstm(packed_input)
#         outputs, output_lengths = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
#         hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
#         output = self.fc(hidden)
#         return output


def yield_tokens(data):
    for tweet, _ in data:
        yield tokenizer(tweet)


def load_and_preprocess_data(train_or_test,argument):
    data = pd.read_csv(train_or_test)[[argument, 'sentiment']]
    data[argument] = data[argument].fillna('', inplace=False)
    data = data.values.tolist()#[:train_set_size]
    data = [(text, sentiment) for text, sentiment in data]
    return data


def create_vocab(train_data):
    vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def train_model(train_loader, model, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    for epoch in range(num_epochs):
        for tweets, sentiments in train_loader:
            tweets, sentiments = tweets.to(device), sentiments.to(device)
            text_lengths = [len(tweet) for tweet in tweets]
            output = model(tweets)
            #output = model(tweets, text_lengths)
            loss = criterion(output, sentiments)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')


def evaluate_model(test_loader, model,vocab):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for tweets, sentiments in test_loader:
            tweets, sentiments = tweets.to(device), sentiments.to(device)
            output = model(tweets)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().tolist())
            true_labels.extend(sentiments.cpu().tolist())
    correct = sum(pred == true for pred, true in zip(predictions, true_labels))
    accuracy = correct / len(predictions)
    print(f'Test Accuracy: {accuracy:.4f}')
    return predictions

# def evaluate_model(test_loader, model, vocab, train_dataset):
#     model.eval()
#     predictions = []
#     true_labels = []
#     with torch.no_grad():
#         for tweet, sentiment in test_loader:
#             print("tweet ",tweet)
#             print("sentiment ",sentiment)
#             # text_lengths = [len(tweet) for tweet in tweets]
#             output = model(tweet)
#             #output = model(tweets, text_lengths)
#             _, predicted = torch.max(output, 1)
#             predictions.extend(predicted.tolist())
#             true_labels.extend(sentiments.tolist())
#             # # print(tweet, true_label)
#             # tweet_ids = vocab(tokenizer(tweet)) + [train_dataset.pad_idx] * (train_dataset.max_len - len(vocab(tokenizer(tweet))))
#             # text_lengths = [len(tweet_ids)]
#             # output = model(tweet)
#             # print(output)
#             # #output = model(torch.tensor(tweet_ids).unsqueeze(0), text_lengths)
#             # _, predicted = torch.max(output, 1)
#             # predictions.append(predicted.item())
#             # true_labels.append(SENTIMENT_TO_IDX[true_label])
#     correct = sum(pred == true for pred, true in zip(predictions, true_labels))
#     accuracy = correct / len(predictions)
#     print(f'Test Accuracy: {accuracy:.4f}')
#     return predictions


def save_predictions(predictions, test_data):
    submission = pd.DataFrame({'textID': test_data.index, 'selected_text': predictions})
    submission.to_csv('submission.csv', index=False)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load and preprocess data
    train_data = load_and_preprocess_data('train.csv','selected_text')
    vocab = create_vocab(train_data)

    # Create dataset and data loader
    train_dataset = TweetDataset(train_data, vocab)
    #for tweet_tensor, sentiment_tensor in train_dataset:
        #print("train Tweet Tensor:", tweet_tensor)
        #print("train Sentiment Tensor:", sentiment_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    max_len = train_dataset.max_len

    # Initialize model and send to device
    vocab_size = len(vocab)
    output_dim = len(SENTIMENT_LABELS)
    model = SentimentNN(vocab_size, embedding_dim, hidden_dim, output_dim, max_len).to(device)
    # model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
 
    # Train the model
    train_model(train_loader, model, num_epochs)

    # Load test data
    test_data = load_and_preprocess_data('test.csv','text')
    print(test_data[:5])
    test_dataset = TweetDataset(test_data, vocab)
    # for tweet_tensor, sentiment_tensor in test_dataset:
    #     print("test Tweet Tensor:", tweet_tensor)
    #     print("test Sentiment Tensor:", sentiment_tensor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Evaluate the model
    predictions = evaluate_model(test_data, model, vocab)

    # Save predictions
    #save_predictions(predictions, test_data)


if __name__ == '__main__':
    main()
