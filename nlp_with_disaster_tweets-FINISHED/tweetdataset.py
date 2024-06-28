from torch.utils.data import Dataset
import re
from nltk.tokenize import word_tokenize
import torch

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

tokenizer = word_tokenize


class TweetTrainDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        self.pad_idx = vocab['<pad>']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet, target_idx = self.data[idx]
        tweet = remove_urls(tweet)
        tokenized_tweet = tokenizer(tweet) 
        tokenized_tweet = [item.lower() for item in tokenized_tweet]
        indexed_tweet = [self.vocab[word] for word in tokenized_tweet]
        #indexed_tweet = truncate_sequences(indexed_tweet, max_length)
        tweet_tensor = torch.tensor(indexed_tweet, dtype=torch.long)
        # print("Train Tweet:", tweet)
        # print("Tokenized tweet:", tokenized_tweet)
        # print("Indexed tweet:", indexed_tweet)
        return tweet_tensor, target_idx

class TweetTestDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        self.pad_idx = vocab['<pad>']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id, tweet = self.data[idx]
        tweet = remove_urls(tweet)
        tokenized_tweet = tokenizer(tweet) 
        tokenized_tweet = [item.lower() for item in tokenized_tweet]
        indexed_tweet = [self.vocab[word] for word in tokenized_tweet]
        #indexed_tweet = truncate_sequences(indexed_tweet, max_length)
        tweet_tensor = torch.tensor(indexed_tweet, dtype=torch.long)
        # print("Test Tweet:", tweet)
        # print("Tokenized tweet:", tokenized_tweet)
        # print("Indexed tweet:", indexed_tweet)
        # print("Tweet tensor:", tweet_tensor, "and its type:", type(tweet_tensor))
        return tweet_tensor