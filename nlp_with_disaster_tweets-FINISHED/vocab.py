import re
from collections import defaultdict

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_count = defaultdict(int)

    def add_word(self, word):
        cleaned_word = self.clean_word(word)
        if cleaned_word and self.is_valid_word(cleaned_word):
            if cleaned_word not in self.word2idx:
                index = len(self.word2idx)
                self.word2idx[cleaned_word] = index
                self.idx2word[index] = cleaned_word
            self.word_count[cleaned_word] += 1
    
    def clean_word(self, word):
        # Remove periods, single quotes, and non-ASCII characters
        word = re.sub(r'[^\w\s-]', '', word)
        word = re.sub(r'[^\x00-\x7F]+', '', word)
        return word.strip().lower()

    def is_valid_word(self, word):
        # Check for unwanted patterns
        if re.match(r'^(//t\.co/|.*\d.*|[0-9:]+$)', word):
            return False
        if len(word) < 2:  # Exclude single-character words
            return False
        return True

    def __getitem__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)

    def __iter__(self):
        return iter(self.word2idx.items())
