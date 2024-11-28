import os
import random

def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.dirname(script_dir)

def load_fasttext_dataset(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        reviews = f.readlines()
    
    reviews = [review.strip() for review in reviews]
    
    return reviews

def split_dataset(reviews, train_ratio=0.75, test_ratio=0.15, validation_ratio=0.10):
    assert abs(train_ratio + test_ratio + validation_ratio - 1.0) < 1e-10, \
        "Ratios must sum to 1.0"
    
    random.seed(42)
    random.shuffle(reviews)
    
    total_reviews = len(reviews)
    train_end = int(total_reviews * train_ratio)
    test_end = train_end + int(total_reviews * test_ratio)
    
    train_set = reviews[:train_end]
    test_set = reviews[train_end:test_end]
    validation_set = reviews[test_end:]
    
    return train_set, test_set, validation_set

def save_dataset(dataset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for review in dataset:
            f.write(review + '\n')
    
    print(f"Dataset saved to {output_file}")
    print(f"Total reviews: {len(dataset)}")
    
    positive_count = sum(1 for review in dataset if '__label__positive' in review)
    negative_count = sum(1 for review in dataset if '__label__negative' in review)
    print(f"Positive reviews: {positive_count}")
    print(f"Negative reviews: {negative_count}")

def main():
    project_root = get_project_root()
    
    input_file = os.path.join(project_root, 'Datasets', 'imdb_movie.txt')
    output_dir = os.path.join(project_root, 'Datasets')
    
    train_output = os.path.join(output_dir, 'imdb_train.txt')
    test_output = os.path.join(output_dir, 'imdb_test.txt')
    validation_output = os.path.join(output_dir, 'imdb_validation.txt')
    
    reviews = load_fasttext_dataset(input_file)
    
    train_set, test_set, validation_set = split_dataset(
        reviews, 
        train_ratio=0.75, 
        test_ratio=0.15, 
        validation_ratio=0.10
    )
    
    save_dataset(train_set, train_output)
    save_dataset(test_set, test_output)
    save_dataset(validation_set, validation_output)

if __name__ == '__main__':
    main()