from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Load model
model_name = "sandbox338/hatespeech"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Political keywords for context detection
POLITICAL_KEYWORDS = ["mheshimiwa", "bunge", "serikali", "waziri", "rais", "kura"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '').lower().strip()
        context = data.get('context', 'all')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Check if text matches selected context
        is_political = any(keyword in text for keyword in POLITICAL_KEYWORDS)
        
        # Context mismatch warning
        warning = ""
        if context == 'political' and not is_political:
            warning = "Warning: Non-political text detected in political mode"
        elif context == 'non-political' and is_political:
            warning = "Warning: Political text detected in non-political mode"

        # Get predictions
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
        
        return jsonify({
            "text": text,
            "probabilities": {
                "non_hate": probs[0],
                "political_hate": probs[1],
                "offensive": probs[2]
            },
            "predicted_class": int(torch.argmax(outputs.logits, dim=1)),
            "is_political": is_political,
            "warning": warning
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)