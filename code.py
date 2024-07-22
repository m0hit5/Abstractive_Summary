!pip install transformers[torch]
!pip install requests
!pip install rouge
!pip install --upgrade transformers

import requests
import torch
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer
)
from rouge import Rouge
import re
import json

# Initialize ROUGE scorer
rouge = Rouge()

# Initialize transformers models and tokenizers
models = {
    "T5": (T5ForConditionalGeneration.from_pretrained("t5-small"), T5Tokenizer.from_pretrained("t5-small")),
    "BART": (BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn"), BartTokenizer.from_pretrained("facebook/bart-large-cnn")),
    "Pegasus": (PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum"), PegasusTokenizer.from_pretrained("google/pegasus-xsum"))
}

def preprocess_text(text):
    """Clean and preprocess the input text."""
    # Remove extra spaces
    text = ' '.join(text.split())
    # Remove non-alphanumeric characters (except for punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text

def fetch_data():
    url = "https://datasets-server.huggingface.co/rows?dataset=ccdv%2Fpubmed-summarization&config=document&split=train&offset=0&length=100"
    response = requests.get(url)
    data = response.json()

    articles = []
    for row_data in data['rows']:
        row = row_data['row']
        articles.append({
            "article": preprocess_text(row['article']),
            "abstract": preprocess_text(row.get('abstract', ''))
        })

    return articles

def summarize_and_evaluate(article, abstract):
    summaries = {}
    for name, (model, tokenizer) in models.items():
        # Preprocess article
        try:
            input_ids = tokenizer.encode(article, return_tensors="pt", max_length=1024, truncation=True)
        except Exception as e:
            print(f"Error encoding input for {name}: {e}")
            summaries[name] = ("", 0.0)
            continue

        # Generate summary
        with torch.no_grad():
            try:
                generated_ids = model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, min_length=30, no_repeat_ngram_size=2)
                generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                # Calculate ROUGE scores
                rouge_scores = rouge.get_scores(generated_summary, abstract)
                
                # Apply smoothing to ROUGE scores
                for score in rouge_scores:
                    for key in score:
                        score[key]['f'] = max(score[key]['f'], 1e-10)  # Avoid zero division errors

                rouge_score = rouge_scores[0]['rouge-1']['f']
                summaries[name] = (generated_summary, rouge_score)
            except IndexError as e:
                print(f"IndexError generating summary with {name}: {e}")
                summaries[name] = ("", 0.0)
            except Exception as e:
                print(f"Unexpected error generating summary with {name}: {e}")
                summaries[name] = ("", 0.0)
    return summaries

def main():
    articles = fetch_data()

    for article in articles:
        print(f"Original Abstract: {article['abstract']}\n")
        summaries = summarize_and_evaluate(article['article'], article['abstract'])
        
        for model_name, (summary, score) in summaries.items():
            print(f"{model_name} Generated Summary:\n{summary}")
            print(f"ROUGE Score (F1): {score:.4f}\n")

if __name__ == "__main__":
    main()