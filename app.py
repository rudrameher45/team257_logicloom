# from flask import Flask, request, render_template
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
#
# # Initialize Flask app
# app = Flask(__name__)
#
# # Load the fine-tuned model and tokenizer
# model_dir = "./finetune_model"
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
# model.eval()  # Set model to evaluation mode
#
# # Generate headline from article
# def generate_headline(article, max_length=128, num_beams=5):
#     # Tokenize the input article
#     inputs = tokenizer(article, max_length=256, truncation=True, return_tensors="pt", padding="max_length")
#
#     # Move inputs to the same device as the model
#     input_ids = inputs['input_ids'].to(model.device)
#     attention_mask = inputs['attention_mask'].to(model.device)
#
#     # Generate headline
#     outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=num_beams, early_stopping=True)
#     headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#     return headline
#
# # Home route to render the form and handle POST requests
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     headline = None
#     if request.method == 'POST':
#         article = request.form.get('article')
#         if article:
#             headline = generate_headline(article)
#     return render_template('index.html', headline=headline)
#
# # Run the app
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)


from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_dir = "./finetune_model"  # Replace with your model's directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

def generate_headline(article):
    """
    Generate a headline from the given article using the fine-tuned model.
    """
    inputs = tokenizer.encode(article, return_tensors="pt", max_length=300, truncation=True)
    outputs = model.generate(inputs, max_length=128, num_beams=5, early_stopping=True)
    headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return headline

@app.route("/", methods=["GET", "POST"])
def home():
    headline = None
    if request.method == "POST":
        article = request.form.get("article")
        if article:
            headline = generate_headline(article)
    return render_template("index.html", headline=headline)

if __name__ == "__main__":
    app.run(debug=True)
