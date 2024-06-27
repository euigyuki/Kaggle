import pandas as pd
from vocab import Vocabulary
from nltk.tokenize import word_tokenize
from tweetdataset import TweetTrainDataset, TweetTestDataset
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from config import config
import numpy as np
from models import EnhancedAttentionBiLSTM
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import re
from sklearn.metrics import accuracy_score
import torch.optim as optim
from early_stopping import EarlyStopping

CHECKING_TRAIN_SIZE = False
CHECKING_TEST_SIZE = False
print_statements = config["debugging"]["print_statements"]
tokenizer = word_tokenize
num_epochs = config['training']['num_epochs']
embedding_dim = config['training']['embedding_dim']
hidden_dim = config['training']['hidden_dim']
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
num_layers = config['model']['num_layers']
dropout_rate = config['model']['dropout_rate']
patience_number = config['training']['patience_number']
weight_decay = config['training']['weight_decay']
max_norm = config['training']['max_norm']
DISASTER_LABELS = ['no_disaster', 'disaster']
DISASTER_TO_IDX = {label: idx for idx, label in enumerate(DISASTER_LABELS)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_collate_fn(batch):
    tweets, sentiments = zip(*batch)
    padded_tweets = pad_sequence(tweets, batch_first=True, padding_value=1)
    sentiments = torch.tensor(sentiments)
    return padded_tweets, sentiments

def test_collate_fn(batch):
    padded_tweets = pad_sequence(batch, batch_first=True, padding_value=1)
    return padded_tweets

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

def train_model(train_loader, val_loader, model):
    best_val_accuracy = 0
    best_val_loss = float('inf')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience_number)
    early_stopping = EarlyStopping(patience=patience_number, verbose=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for tweets, sentiments in train_loader:
            tweets, sentiments = tweets.to(device), sentiments.to(device)
            
            optimizer.zero_grad()
            output = model(tweets)
            loss = criterion(output, sentiments)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            
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
        

        # Update best validation metrics
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    model.best_val_accuracy = best_val_accuracy
    model.best_val_loss = best_val_loss
    return model

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


def load_data():
    train_data = load_and_preprocess_train_data()
    test_data = load_and_preprocess_test_data()
    return train_data, test_data

def prepare_datasets(train_data):
    vocab = create_vocab(train_data)
    full_train_dataset = TweetTrainDataset(train_data, vocab)
    
    total_size = len(full_train_dataset)
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    return train_dataset, val_dataset, vocab

def create_data_loaders(train_dataset, val_dataset, test_data, vocab):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)
    
    test_dataset = TweetTestDataset(test_data, vocab)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_collate_fn)
    
    return train_loader, val_loader, test_loader

def setup_model(vocab_size,vocab):
    output_dim = len(DISASTER_LABELS)
    glove_file_path = f'glove/glove.6B.{embedding_dim}d.txt'
    glove_embeddings = load_glove_embeddings(glove_file_path)
    
    model = EnhancedAttentionBiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, 
                                    glove_embeddings, vocab, n_layers=num_layers, 
                                    dropout_rate=dropout_rate).to(device)
    return model

def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

