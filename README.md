# Text-Sentiment-Analysis
Designing a NLP Text Sentiment Analysis Model

## Languages Used
Python

To Download Python, click [here](https://www.python.org/downloads/)

## Model Used
FastText Model

## Dataset Used
IMDB Movie Reviews Dataset

Dataset can be found [here](https://ai.stanford.edu/~amaas/data/sentiment/)

Open-Source, free, and lightweight texNLP model for text classification

## Install Necessary Libraries
Make sure pip is installed

Instructions can be found [here](https://pip.pypa.io/en/stable/installation/)

These Libraries can be downloaded through command line prompts

Install FastText Library:
```
pip install fasttext
```

Install Pandas Library:
```
pip install pandas
```

Install NLTK Library:
```
pip install nltk
```

Install Argparse Library:
```
pip install argparse
```

## Preprocess The Dataset
- **Load Data**
  - Read the text files into pandas DataFrames for positive and negative reviews
- **Clean The Text**
  - Remove special characters
  - Remove extra spaces
  - Remove stopwrods
  - Convert text to lowercase
- **Format Text for FastText Model**
  - FastText Requires Labels:
    - Negative - __label__negative
    - Positive - __label__positive

## Split The Dataset
Split Data into Training, Testing, and Validation
- **Training** - 75% of Dataset
- **Testing** - 15% of Dataset
- **Validation** - 10% of Dataset

## Training the FastText Model
Use FastText Library to train the model
**Key Parameters of the Model are:**
- **Input** - Path to training data file
- **Epoch** - Number of training iterations
- **Lr** - Learning rate
- **WordNGrams** - Number of N-Grams for model to consider

## Evaluate the FastText Model
Test FastText Model's performance on test set
**Performace Metrics Used:**
- Precision
- Recall

## Predict Sentiments:
Use trained model to predict sentiments for the validation dataset

## Fine-Tune FastText Model and Optimize
Adjust hyperparameters for better accuracy:
- Epoch
- Lr
- WordNgrams

Perform Cross-Validation to ensure the FastText Model Generalizes Well

## Save and Load the FastText Model
Save the FastText Model and load when needed