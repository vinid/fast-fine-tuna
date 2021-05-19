from transformers import AutoModel
from torch import nn

class MiniModel(nn.Module):
    def __init__(self, model_name, n_labels_A, n_labels_B):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

        self.first_classifier = nn.Linear(768, n_labels_A)
        self.second_classifier = nn.Linear(768, n_labels_B)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)

        A_logits = self.first_classifier(output[1])
        B_logits = self.second_classifier(output[1])

        return A_logits, B_logits
