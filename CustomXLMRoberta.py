from transformers import XLMRobertaModel
import torch.nn as nn



# ================================
# Model Definition
# ================================
class CustomXLMRoberta(nn.Module):
    def __init__(self, model_name, num_laws_labels):
        super(CustomXLMRoberta, self).__init__()
        self.base_model = XLMRobertaModel.from_pretrained(model_name, output_attentions=True)
        self.dictum_classifier = nn.Linear(self.base_model.config.hidden_size, 2)
        self.laws_classifier = nn.Linear(self.base_model.config.hidden_size, num_laws_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token representation
        dictum_logits = self.dictum_classifier(cls_output)
        laws_logits = self.laws_classifier(cls_output)
        return dictum_logits, laws_logits, outputs.attentions
