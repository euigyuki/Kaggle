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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np

# Training parameters
tokenizer = word_tokenize

num_epochs = 1000
embedding_dim = 200
hidden_dim = 128
learning_rate = 0.0001
batch_size = 64
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
patience_number = 10


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

def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index



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





class EnhancedAttentionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,word2vec_model,vocab, n_layers=2, dropout_rate=0.3):
        super(EnhancedAttentionBiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.initialize_embeddings(word2vec_model, vocab)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=True, 
                            batch_first=True, 
                            dropout=dropout_rate if n_layers > 1 else 0)
        
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)


    def initialize_embeddings(self, glove_embeddings, vocab):
        for word, idx in vocab.word2idx.items():
            if word in glove_embeddings:
                self.embedding.weight.data[idx] = torch.FloatTensor(glove_embeddings[word])
            else:
                self.embedding.weight.data[idx] = torch.FloatTensor(embedding_dim).uniform_(-0.25, 0.25)



    def attention_mechanism(self, lstm_output):
        attn_weights = torch.tanh(self.attention(lstm_output))
        attn_weights = torch.sum(attn_weights, dim=2)
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(2)
        context = torch.bmm(lstm_output.transpose(1, 2), attn_weights).squeeze(2)
        return context

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Attention mechanism
        attn_output = self.attention_mechanism(lstm_output)
        
        # Fully connected layers with dropout and batch normalization
        x = self.dropout(attn_output)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class EarlyStopping:
    def __init__(self, patience=patience_number, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def train_model(train_loader, val_loader, model, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    early_stopping = EarlyStopping(patience=5, verbose=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for tweets, sentiments in train_loader:
            analyze_padding(tweets)
            tweets, sentiments = tweets.to(device), sentiments.to(device)
            
            optimizer.zero_grad()
            output = model(tweets)
            loss = criterion(output, sentiments)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_true_labels = []
        correct_predictions = 0
        with torch.no_grad():
            for tweets, sentiments in val_loader:
                tweets, sentiments = tweets.to(device), sentiments.to(device)
                output = model(tweets)
                loss = criterion(output, sentiments)
                val_loss += loss.item()
        
                _, predicted = torch.max(output, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(sentiments.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

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
    full_train_dataset = TweetTrainDataset(train_data, vocab)
    test_dataset = TweetTestDataset(test_data, vocab)

    total_size = len(full_train_dataset)
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_collate_fn)

    vocab_size = len(vocab)
    output_dim = len(DISASTER_LABELS)
    
    glove_file_path = 'glove/glove.6B.200d.txt'
    glove_embeddings = load_glove_embeddings(glove_file_path)
        
    model = EnhancedAttentionBiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, glove_embeddings, vocab, n_layers=2, dropout_rate=0.3).to(device)


    #model = EnhancedAttentionBiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout_rate=0.3).to(device)
 
    train_model(train_loader, val_loader, model, num_epochs)

    predictions = evaluate_model(test_loader, model)
    create_submission(predictions, test_data, 'final_submission.csv')

    evaluate_model(test_loader, model)
    print("Average padding percentage:", sum(paddings) / len(paddings))

if __name__ == '__main__':
    main()