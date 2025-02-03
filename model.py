import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class ChatbotModel(nn.Module):
    def __init__(self, vocab_size):
        super(ChatbotModel, self).__init__()
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=768,
            n_layer=12,  
            n_head=12,  
            bos_token_id=0,
            eos_token_id=1
        )
        self.model = GPT2LMHeadModel(self.config)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask).logits

    @classmethod
    def from_pretrained(cls, checkpoint_path, vocab_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(vocab_size).to(device)
        model.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model