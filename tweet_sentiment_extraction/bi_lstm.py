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
batch_size = 4
train_set_size = 4

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
        tweet_ids = self.vocab(tokenizer(tweet)) + [self.pad_idx] * (self.max_len - len(self.vocab(tokenizer(tweet))))
        sentiment_idx = SENTIMENT_TO_IDX[sentiment]
        tweet_tensor = torch.tensor(tweet_ids, dtype=torch.long)
        return tweet_tensor, sentiment_idx


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text, text_lengths):
        embeddings = self.embedding(text)
        packed_input = rnn_utils.pack_padded_sequence(embeddings, text_lengths, batch_first=True, enforce_sorted=False)
        outputs, (hidden, _) = self.lstm(packed_input)
        outputs, output_lengths = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.fc(hidden)
        return output


def yield_tokens(data):
    for tweet, _ in data:
        yield tokenizer(tweet)


def load_and_preprocess_data():
    train_data = pd.read_csv('train.csv')[['selected_text', 'sentiment']]
    train_data['selected_text'] = train_data['selected_text'].fillna('', inplace=False)
    train_data = train_data.values.tolist()[:train_set_size]
    print(train_data)
    train_data = [(text, sentiment) for text, sentiment in train_data]
    return train_data


def create_vocab(train_data):
    vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def train_model(train_loader, model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for tweets, sentiments in train_loader:
            print(tweets, sentiments)
            text_lengths = [len(tweet) for tweet in tweets]
            output = model(tweets, text_lengths)
            loss = criterion(output, sentiments)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')


def evaluate_model(test_data, model, vocab, train_dataset):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for idx in range(len(test_data)):
            tweet = test_data['text'][idx]
            true_label = test_data['sentiment'][idx]
            print(tweet, true_label)
            tweet_ids = vocab(tokenizer(tweet)) + [train_dataset.pad_idx] * (train_dataset.max_len - len(vocab(tokenizer(tweet))))
            text_lengths = [len(tweet_ids)]
            output = model(torch.tensor(tweet_ids).unsqueeze(0), text_lengths)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())
            true_labels.append(SENTIMENT_TO_IDX[true_label])

    correct = sum(pred == true for pred, true in zip(predictions, true_labels))
    accuracy = correct / len(predictions)
    print(f'Test Accuracy: {accuracy:.4f}')

    return predictions


def save_predictions(predictions, test_data):
    submission = pd.DataFrame({'textID': test_data.index, 'selected_text': predictions})
    submission.to_csv('submission.csv', index=False)


def main():
    # Load and preprocess data
    train_data = load_and_preprocess_data()
    vocab = create_vocab(train_data)

    # Create dataset and data loader
    train_dataset = TweetDataset(train_data, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    vocab_size = len(vocab)
    output_dim = len(SENTIMENT_LABELS)
    print("sentiment labels: ",SENTIMENT_LABELS)
    print(vocab["have"])
    print(vocab["leave"])
    model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(train_loader, model, criterion, optimizer, num_epochs)

    # Load test data
    test_data = pd.read_csv('test.csv')
    
    # Evaluate the model
    #predictions = evaluate_model(test_data, model, vocab, train_dataset)

    # Save predictions
    #save_predictions(predictions, test_data)


if __name__ == '__main__':
    main()
