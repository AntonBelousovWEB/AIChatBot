import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import ChatbotModel
from dataset_loader import load_dataset
import json
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

VOCAB_SIZE = config["vocab_size"]
EMBED_DIM = config["embed_dim"]
HIDDEN_DIM = config["hidden_dim"]
NUM_LAYERS = config["num_layers"]
EPOCHS = config["epochs"]
BATCH_SIZE = config["batch_size"]
NUM_WORKERS = config["num_workers"]
CHECKPOINT_PATH = config["checkpoint_path"]

def create_optimizer(model, lr=0.001):
    return optim.Adam(model.parameters(), lr=lr)

if __name__ == "__main__":
    print("🔄 Загрузка датасета...")
    dataset, vocab = load_dataset("dataset.csv", VOCAB_SIZE)
    print(f"✅ Датасет загружен! Найдено {len(vocab)} уникальных слов.")

    os.makedirs("trained_model", exist_ok=True)
    with open("trained_model/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print("✅ Вокабуляр сохранён!")

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"✅ DataLoader создан! {len(train_loader)} батчей для обучения.")

    start_epoch = 0
    start_batch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Найдена контрольная точка {CHECKPOINT_PATH}, загружаем...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
        old_vocab_size = checkpoint["model_state"]["embedding.weight"].shape[0]
        
        if old_vocab_size < VOCAB_SIZE:
            print(f"🔄 Расширяем модель с {old_vocab_size} до {VOCAB_SIZE} слов...")
            model = ChatbotModel.from_pretrained(checkpoint["model_state"], new_vocab_size=VOCAB_SIZE)
            optimizer = create_optimizer(model)
        else:
            model = ChatbotModel.from_pretrained(checkpoint["model_state"])
            optimizer = create_optimizer(model)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint["batch"]
        print(f"✅ Обучение продолжается с эпохи {start_epoch + 1}, батча {start_batch}.")
    else:
        print("⚠️ Контрольная точка не найдена. Начинаем обучение с нуля.")
        model = ChatbotModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS)
        optimizer = create_optimizer(model)

    print("🚀 Начинаем обучение...")
    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx < start_batch:
                continue

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"🟢 Эпоха {epoch+1}/{EPOCHS}, Батч {batch_idx}/{len(train_loader)}, Потери: {loss.item():.4f}")

            if batch_idx % 50 == 0:
                torch.save({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, CHECKPOINT_PATH)
                print(f"💾 Прогресс сохранён! (эпоха {epoch+1}, батч {batch_idx})")

        print(f"✅ Эпоха {epoch+1} завершена! Средние потери: {epoch_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "trained_model/chatbot.pth")
    print("✅ Финальная модель сохранена!")

    print("🎉 Обучение завершено!")
    os.remove(CHECKPOINT_PATH)