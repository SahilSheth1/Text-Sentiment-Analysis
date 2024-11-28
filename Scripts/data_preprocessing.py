import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

def load_imdb_reviews(dataset_path):
    def read_reviews_from_folder(folder_path):
        reviews = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        reviews.append(file.read().strip())
                except Exception as e:
                    print(f"Error reading file {filepath}: {e}")
        return reviews
    
    train_pos_path = os.path.join(dataset_path, 'train', 'pos')
    train_neg_path = os.path.join(dataset_path, 'train', 'neg')
    test_pos_path = os.path.join(dataset_path, 'test', 'pos')
    test_neg_path = os.path.join(dataset_path, 'test', 'neg')
    
    train_pos_reviews = read_reviews_from_folder(train_pos_path)
    train_neg_reviews = read_reviews_from_folder(train_neg_path)
    test_pos_reviews = read_reviews_from_folder(test_pos_path)
    test_neg_reviews = read_reviews_from_folder(test_neg_path)
    
    all_pos_reviews = train_pos_reviews + test_pos_reviews
    all_neg_reviews = train_neg_reviews + test_neg_reviews
    
    return all_pos_reviews, all_neg_reviews

def clean_text(text):
    text = text.lower()
    
    text = re.sub(r'<[^>]+>', '', text)
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

def preprocess_reviews(positive_reviews, negative_reviews):
    cleaned_pos_reviews = [clean_text(review) for review in positive_reviews]
    cleaned_neg_reviews = [clean_text(review) for review in negative_reviews]
    
    positive_reviews_fasttext = [f"__label__positive {review}" for review in cleaned_pos_reviews]
    negative_reviews_fasttext = [f"__label__negative {review}" for review in cleaned_neg_reviews]
    
    return positive_reviews_fasttext, negative_reviews_fasttext

def save_fasttext_data(positive_reviews, negative_reviews, output_file):
    import random
    
    all_reviews = positive_reviews + negative_reviews
    random.shuffle(all_reviews)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for review in all_reviews:
            f.write(review + '\n')
    
    print(f"Preprocessed data saved to {output_file}")
    print(f"Total reviews processed: {len(all_reviews)}")
    print(f"Positive reviews: {len(positive_reviews)}")
    print(f"Negative reviews: {len(negative_reviews)}")

def main():
    DATASET_PATH = 'IMDBMovieDataset'
    OUTPUT_FILE = 'Datasets/imdb_movie.txt'
    
    positive_reviews, negative_reviews = load_imdb_reviews(DATASET_PATH)
    
    positive_fasttext, negative_fasttext = preprocess_reviews(positive_reviews, negative_reviews)
    
    save_fasttext_data(positive_fasttext, negative_fasttext, OUTPUT_FILE)

if __name__ == '__main__':
    main()