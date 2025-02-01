import torch.nn as nn

class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(ChatbotModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output)
        return output

    def expand_embedding(self, new_vocab_size):
        old_vocab_size, embed_dim = self.embedding.weight.shape
        if new_vocab_size <= old_vocab_size:
            return

        new_embedding = nn.Embedding(new_vocab_size, embed_dim)
        new_embedding.weight.data[:old_vocab_size] = self.embedding.weight.data
        self.embedding = new_embedding

        new_fc = nn.Linear(self.fc.in_features, new_vocab_size)
        new_fc.weight.data[:old_vocab_size, :] = self.fc.weight.data
        new_fc.bias.data[:old_vocab_size] = self.fc.bias.data
        self.fc = new_fc

    @classmethod
    def from_pretrained(cls, state_dict, new_vocab_size=None):
        old_vocab_size = state_dict['embedding.weight'].shape[0]
        embed_dim = state_dict['embedding.weight'].shape[1]
        hidden_dim = state_dict['lstm.weight_hh_l0'].shape[0] // 4
        num_layers = len([k for k in state_dict.keys() if k.startswith('lstm.weight_hh_l')])

        model = cls(old_vocab_size, embed_dim, hidden_dim, num_layers)
        model.load_state_dict(state_dict)

        if new_vocab_size and new_vocab_size > old_vocab_size:
            model.expand_embedding(new_vocab_size)

        return model