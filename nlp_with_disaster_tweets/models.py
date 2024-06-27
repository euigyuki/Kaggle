import torch.nn as nn
from config import config
import torch
import torch.nn.functional as F


num_layers = config['model']['num_layers']
dropout_rate = config['model']['dropout_rate']
embedding_dim = config['training']['embedding_dim']



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

class EnhancedAttentionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,embedding_model,vocab, n_layers=num_layers, dropout_rate=dropout_rate):
        super(EnhancedAttentionBiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.initialize_embeddings(embedding_model, vocab)
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
