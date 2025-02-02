import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import ChatbotModel
from dataset_loader import load_dataset
import json
import os
import warnings
import torch.backends.cudnn as cudnn

warnings.filterwarnings("ignore", category=RuntimeWarning)

cudnn.benchmark = True

def main():
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
    VOCAB_PATH = "trained_model/vocab.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Используемое устройство: {device}")

    if os.path.exists(VOCAB_PATH):
        print("✅ Найден существующий словарь, загружаем...")
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        dataset, _ = load_dataset("dataset.csv", VOCAB_SIZE)
    else:
        print("🔄 Словарь не найден, создаём новый...")
        dataset, vocab = load_dataset("dataset.csv", VOCAB_SIZE)
        os.makedirs("trained_model", exist_ok=True)
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)
        print("✅ Новый словарь сохранён!")

    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    print(f"✅ DataLoader создан! {len(train_loader)} батчей для обучения.")

    model = ChatbotModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch, start_batch = 0, 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Найдена контрольная точка {CHECKPOINT_PATH}, загружаем...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch, start_batch = checkpoint["epoch"], checkpoint["batch"]
        print(f"✅ Обучение продолжается с эпохи {start_epoch + 1}, батча {start_batch}.")

    print("🚀 Начинаем обучение...")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = F.cross_entropy(outputs.view(-1, VOCAB_SIZE), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"🟢 Эпоха {epoch+1}/{EPOCHS}, Батч {batch_idx}/{len(train_loader)}, Лосс: {loss.item():.4f}")

            if batch_idx % 50 == 0:
                torch.save({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, CHECKPOINT_PATH)
                print(f"💾 Прогресс сохранён! (эпоха {epoch+1}, батч {batch_idx})")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Эпоха {epoch+1} завершена! Средний лосс: {avg_loss:.4f}")

    torch.save(model.state_dict(), "trained_model/chatbot.pth")
    print("✅ Финальная модель сохранена!")
    print("🎉 Обучение завершено!")

if __name__ == "__main__":
    main()