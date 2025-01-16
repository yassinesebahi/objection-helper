from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import torch
import torch.nn.functional as F

# ================================
# Set device (GPU if available, otherwise CPU)
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# Load XLM-RoBERTa model and tokenizer
# ================================
model = XLMRobertaForSequenceClassification.from_pretrained("XLM-RoBERTa.bin")
tokenizer = XLMRobertaTokenizer.from_pretrained("XLMRobertaTokenizer.json")

model.to(device)

# ================================
# Predict with the loaded model
# ================================

new_objection = "It is important to adjust and make it better for the current situation."

# Tokenize and predict using the loaded model
def predict_with_loaded_model(text):
    """
    Predicts the label for a given text using the loaded model and tokenizer.
    """
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get probabilities using softmax
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get predicted label (class with highest probability)
        predicted_label = torch.argmax(torch.tensor(probabilities)).item()
        
        return predicted_label, probabilities

# Test the model with a new objection
predicted_label, predicted_probabilities = predict_with_loaded_model(new_objection)
print(f"Predicted label: {predicted_label} (1: Gegrond, 0: Ongegrond)")
print(f"Probabity (Ongegrond): {predicted_probabilities[0]}\nProbabity (Gegrond): {predicted_probabilities[1]}")
