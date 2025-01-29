import pandas as pd
import torch
from torch.utils.data import Dataset
import nltk
from nltk.tokenize import word_tokenize
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

if not os.path.exists(os.path.join(nltk.data.find('tokenizers'), 'punkt.zip')):
    nltk.download('punkt', quiet=True)

class TextDataset(Dataset):
    def __init__(self, dataframe, vocab, seq_length=50):
        self.texts = dataframe["tweet"].dropna().tolist()
        self.vocab = vocab
        self.seq_length = seq_length

        print(f"🔄 Начинаем токенизацию {len(self.texts)} строк...")
        self.processed_texts = self.preprocess_texts()
        print(f"✅ Токенизация завершена! Всего слов после токенизации: {len(self.processed_texts)}")

    def preprocess_texts(self):
        return [word for text in self.texts for word in word_tokenize(text.lower())]

    def __len__(self):
        return len(self.processed_texts) - self.seq_length

    def __getitem__(self, idx):
        input_text = self.processed_texts[idx: idx + self.seq_length]
        target_text = self.processed_texts[idx + 1: idx + self.seq_length + 1]
        input_ids = torch.tensor([self.vocab.get(word, 0) for word in input_text], dtype=torch.long)
        target_ids = torch.tensor([self.vocab.get(word, 0) for word in target_text], dtype=torch.long)
        return input_ids, target_ids

def build_vocab(texts, vocab_size=10000):
    print(f"🔄 Строим словарь из {len(texts)} текстов...")
    words = [word for text in texts for word in word_tokenize(text.lower())]
    freq_dist = nltk.FreqDist(words)
    vocab = {word: i for i, (word, _) in enumerate(freq_dist.most_common(vocab_size))}
    print(f"✅ Словарь создан! Размер: {len(vocab)} слов.")
    return vocab

def load_dataset(csv_path, vocab_size=10000):
    print(f"🔄 Загружаем CSV {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✅ CSV загружен! Всего строк: {len(df)}")
    
    texts = df["tweet"].dropna().tolist()
    vocab = build_vocab(texts, vocab_size)
    dataset = TextDataset(df, vocab)
    return dataset, vocab