
# Abstractive summarization of PubMed articles using Transformers

This project demonstrates how to use transformer-based models for abstractive summarization of scientific articles. The code fetches articles from the PubMed dataset, generates summaries using T5, BART, and Pegasus models, and evaluates the summaries using ROUGE scores.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Description](#project-description)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Acknowledgments](#acknowledgments)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/m0hit5/Abstractive_Summary
    cd abstractive-summarization
    ```

2. Install the required packages:
    ```bash
    pip install transformers[torch]
    pip install requests
    pip install rouge
    pip install --upgrade transformers
    ```

## Usage

1. Run the script:
    ```bash
    python code.py
    ```

2. The script will fetch a few articles from the PubMed dataset, generate summaries using T5, BART, and Pegasus models, and print the original abstracts along with the generated summaries and their respective ROUGE scores.

## Project Description

This project fetches articles and abstracts from the PubMed dataset and performs the following steps:

1. **Preprocesses** the text by cleaning and removing unnecessary characters.
2. **Generates summaries** for each article using three transformer models: T5, BART, and Pegasus.
3. **Calculates ROUGE scores** to evaluate the generated summaries against the original abstracts.
4. **Prints** the original abstracts, generated summaries, and ROUGE scores for comparison.

### Key Components
- **Fetching Data:** Retrieves articles and abstracts from the PubMed dataset.
- **Preprocessing:** Cleans and prepares the text for summarization.
- **Summarization:** Generates summaries using T5, BART, and Pegasus models.
- **Evaluation:** Calculates ROUGE scores to compare generated summaries with original abstracts.

### Hyperparameter Tuning

Hyperparameter tuning can significantly impact the performance of summarization models. In this project, you can tune the following hyperparameters for each model:

- `max_length`: The maximum length of the generated summary.
- `num_beams`: The number of beams for beam search. This controls the diversity of the generated summaries.
- `length_penalty`: Exponential penalty to the length. A value < 1.0 encourages shorter sequences, while > 1.0 encourages longer sequences.
- `min_length`: The minimum length of the generated summary.
- `no_repeat_ngram_size`: The size of the n-grams that should not be repeated in the generated text.

You can modify these parameters in the `summarize_and_evaluate` function.

## Acknowledgments

- This project uses the [transformers](https://github.com/huggingface/transformers) library by Hugging Face.
- The PubMed dataset is provided by [ccdv](https://huggingface.co/datasets/ccdv/pubmed-summarization?viewer_api=true).
