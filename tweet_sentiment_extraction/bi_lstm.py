import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class TweetDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        self.pad_idx = vocab['<pad>']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet, sentiment = self.data[idx]
        if isinstance(tweet, str):
            tweet_ids = self.vocab(tokenizer(tweet))
        else:
            tweet_ids = []
        sentiment_idx = SENTIMENT_TO_IDX[sentiment]
        tweet_tensor = torch.tensor(tweet_ids, dtype=torch.long)
        return tweet_tensor, sentiment_idx

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embeddings = self.embedding(text)
        _, (hidden, _) = self.lstm(embeddings)
        hidden = hidden.squeeze(0)
        output = self.fc(hidden)
        return output

# Define sentiment labels and their indices
SENTIMENT_LABELS = ['positive', 'neutral', 'negative']
SENTIMENT_TO_IDX = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}




# Load and preprocess data
train_data = pd.read_csv('train.csv')[['text', 'sentiment']]
print(train_data.head())
train_data['text'] = train_data['text'].fillna('', inplace=False)
train_data = train_data.values.tolist()
print(train_data)
print()


# Tokenize and create vocabulary
tokenizer = get_tokenizer('basic_english')
def yield_tokens(data):
    for tweet, _ in data:
        yield tokenizer(tweet)

vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=['<unk>', '<pad>'])
vocab.set_default_index(vocab['<unk>'])

# Create dataset and data loader
train_dataset = TweetDataset(train_data, vocab)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)



# Initialize model, loss, and optimizer
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
output_dim = len(SENTIMENT_LABELS)
model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for tweets, sentiments in train_loader:
        outputs = model(tweets)
        loss = criterion(outputs, sentiments)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Inference on test data
test_data = pd.read_csv('test.csv')['text'].tolist()

model.eval()
predictions = []
with torch.no_grad():
    for tweet in test_data:
        tweet_ids = vocab(tokenizer(tweet))
        output = model(torch.tensor(tweet_ids).unsqueeze(0))
        _, predicted = torch.max(output, 1)
        sentiment = SENTIMENT_LABELS[predicted.item()]
        predictions.append(sentiment)

# Save predictions
submission = pd.DataFrame({'textID': test_data.index, 'selected_text': predictions})
submission.to_csv('submission.csv', index=False)