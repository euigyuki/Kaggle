import re
from collections import defaultdict

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_count = defaultdict(int)

    def add_word(self, word):
        if self.is_valid_word(word):
            if word not in self.word2idx:
                index = len(self.word2idx)
                self.word2idx[word] = index
                self.idx2word[index] = word
            self.word_count[word] += 1
    
    def is_valid_word(self, word):
        # Pattern to match t.co links and other unwanted patterns
        pattern  = re.sub(r'[^\x00-\x7F]+', '', word)
        pattern = r'^(//t\.co/|.*\d.*|[0-9:]+$|[^\x00-\x7F]+)'
        return not re.match(pattern, word)

    def __getitem__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)

    def __iter__(self):
        return iter(self.word2idx.items())