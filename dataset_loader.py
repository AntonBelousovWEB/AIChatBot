import pandas as pd
import torch
from torch.utils.data import Dataset
import nltk
from nltk.tokenize import word_tokenize
import os
import warnings
import re
import ast

warnings.filterwarnings("ignore", category=RuntimeWarning)

nltk.download('punkt', quiet=True)

class TextDataset(Dataset):
    def __init__(self, dataframe, vocab, seq_length=100):
        self.texts = self.process_dialogs(dataframe["dialog"].dropna().tolist())
        self.vocab = vocab
        self.seq_length = seq_length

        print(f"🔄 Токенизация {len(self.texts)} реплик...")
        self.processed_texts = self.preprocess_texts()
        print(f"✅ Токенизация завершена! Всего слов: {len(self.processed_texts)}")

    def process_dialogs(self, dialog_list):
        cleaned_dialogs = []
        for raw_text in dialog_list:
            try:
                sentences = ast.literal_eval(raw_text)
                if isinstance(sentences, list):
                    cleaned_dialogs.extend([self.clean_text(sentence) for sentence in sentences])
            except Exception as e:
                print(f"Ошибка обработки диалога: {e}")
        return cleaned_dialogs

    def clean_text(self, text):
        text = text.lower()

        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"<.*?>", "", text)

        text = text.replace("``", "").replace("''", "").replace("[", "").replace("]", "").replace("'", " ")

        text = re.sub(r"(?<=[a-zA-Z])\.(?=[a-zA-Z])", " . ", text)

        tokens = word_tokenize(text)

        return tokens


    def preprocess_texts(self):
        return [word for sentence in self.texts for word in sentence]

    def __len__(self):
        return len(self.processed_texts) - self.seq_length

    def __getitem__(self, idx):
        input_text = self.processed_texts[idx: idx + self.seq_length]
        target_text = self.processed_texts[idx + 1: idx + self.seq_length + 1]
        input_ids = torch.tensor([self.vocab.get(word, 0) for word in input_text], dtype=torch.long)
        target_ids = torch.tensor([self.vocab.get(word, 0) for word in target_text], dtype=torch.long)
        return input_ids, target_ids

def build_vocab(texts, vocab_size=20000):
    print(f"🔄 Строим словарь из {len(texts)} диалогов...")
    words = [word for text in texts for word in TextDataset.clean_text(TextDataset, text)]

    mandatory_tokens = {".", "?", "!", "..."}
    freq_dist = nltk.FreqDist(words)
    
    vocab = {word: i+1 for i, (word, _) in enumerate(freq_dist.most_common(vocab_size - len(mandatory_tokens) - 1))}
    
    for token in mandatory_tokens:
        if token not in vocab:
            vocab[token] = len(vocab) + 1

    vocab['<UNK>'] = 0
    print(f"✅ Словарь создан! Размер: {len(vocab)} слов.")
    return vocab

def load_dataset(csv_path, vocab_size=20000):
    print(f"🔄 Загружаем CSV {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✅ CSV загружен! Всего строк: {len(df)}")

    texts = df["dialog"].dropna().tolist()
    vocab = build_vocab(texts, vocab_size)
    dataset = TextDataset(df, vocab)
    return dataset, vocab