import torch
import json
from nltk.tokenize import word_tokenize
from model import ChatbotModel
import os
import torch.nn.functional as F

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

VOCAB_SIZE = config["vocab_size"]
EMBED_DIM = config["embed_dim"]
HIDDEN_DIM = config["hidden_dim"]
NUM_LAYERS = config["num_layers"]
CHECKPOINT_PATH = config["checkpoint_path"]
FINAL_MODEL_PATH = config["final_model_path"]

with open("trained_model/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

model = ChatbotModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS)

def load_model(checkpoint=False):
    if checkpoint and os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Загружаем модель из чекпоинта {CHECKPOINT_PATH}...")
        checkpoint_data = torch.load(CHECKPOINT_PATH, weights_only=True)
        model.load_state_dict(checkpoint_data["model_state"])
        print("✅ Чекпоинт загружен!")
    elif os.path.exists(FINAL_MODEL_PATH):
        print(f"🔄 Загружаем финальную модель {FINAL_MODEL_PATH}...")
        model.load_state_dict(torch.load(FINAL_MODEL_PATH, weights_only=True))
        print("✅ Финальная модель загружена!")
    else:
        print("⚠️ Модель не найдена! Используется случайная инициализация.")
    
    model.eval()
    return model

def generate_response(input_text, checkpoint=False, max_length=20):
    model = load_model(checkpoint)
    words = word_tokenize(input_text.lower())
    input_ids = torch.tensor([[vocab.get(word, 0) for word in words]], dtype=torch.long)

    response_words = []
    model.eval()

    for _ in range(max_length):
        output = model(input_ids)
        probabilities = F.softmax(output[0, -1], dim=-1)
        predicted_id = torch.multinomial(probabilities, num_samples=1).item()

        if predicted_id in vocab.values():
            next_word = list(vocab.keys())[list(vocab.values()).index(predicted_id)]
        else:
            next_word = "..."

        response_words.append(next_word)
        input_ids = torch.tensor([[vocab.get(word, 0) for word in (words + response_words)[-10:]]], dtype=torch.long)

        if next_word in [".", "!", "?", "..."]:
            break

    return " ".join(response_words)

while True:
    user_input = input("Вы: ")
    if user_input.lower() == "exit":
        break
    use_checkpoint = input("Использовать чекпоинт? (yes/no): ").strip().lower() == "yes"
    print("Бот:", generate_response(user_input, checkpoint=use_checkpoint))