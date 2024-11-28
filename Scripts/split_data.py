import os
import random

def get_project_root():
    """
    Get the absolute path to the project root directory.
    
    Returns:
    --------
    str: Absolute path to the project root directory
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root
    return os.path.dirname(script_dir)

def load_fasttext_dataset(input_file):
    """
    Load the preprocessed FastText dataset.
    
    Parameters:
    -----------
    input_file : str
        Path to the preprocessed FastText format file
    
    Returns:
    --------
    list: Reviews with FastText labels
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        reviews = f.readlines()
    
    # Remove any trailing whitespace
    reviews = [review.strip() for review in reviews]
    
    return reviews

def split_dataset(reviews, train_ratio=0.75, test_ratio=0.15, validation_ratio=0.10):
    """
    Split the dataset into training, test, and validation sets.
    
    Parameters:
    -----------
    reviews : list
        List of reviews with FastText labels
    train_ratio : float, optional
        Proportion of data for training (default: 0.75)
    test_ratio : float, optional
        Proportion of data for testing (default: 0.15)
    validation_ratio : float, optional
        Proportion of data for validation (default: 0.10)
    
    Returns:
    --------
    tuple: (train_set, test_set, validation_set)
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + test_ratio + validation_ratio - 1.0) < 1e-10, \
        "Ratios must sum to 1.0"
    
    # Shuffle the reviews to ensure random distribution
    random.seed(42)  # for reproducibility
    random.shuffle(reviews)
    
    # Calculate split indices
    total_reviews = len(reviews)
    train_end = int(total_reviews * train_ratio)
    test_end = train_end + int(total_reviews * test_ratio)
    
    # Split the dataset
    train_set = reviews[:train_end]
    test_set = reviews[train_end:test_end]
    validation_set = reviews[test_end:]
    
    return train_set, test_set, validation_set

def save_dataset(dataset, output_file):
    """
    Save a dataset to a file in FastText format.
    
    Parameters:
    -----------
    dataset : list
        List of reviews with FastText labels
    output_file : str
        Path to save the dataset
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for review in dataset:
            f.write(review + '\n')
    
    print(f"Dataset saved to {output_file}")
    print(f"Total reviews: {len(dataset)}")
    
    # Count and print label distribution
    positive_count = sum(1 for review in dataset if '__label__positive' in review)
    negative_count = sum(1 for review in dataset if '__label__negative' in review)
    print(f"Positive reviews: {positive_count}")
    print(f"Negative reviews: {negative_count}")

def main():
    # Get project root directory
    project_root = get_project_root()
    
    # Construct input and output file paths
    input_file = os.path.join(project_root, 'Datasets', 'imdb_movie.txt')
    output_dir = os.path.join(project_root, 'Datasets')
    
    # Output file paths
    train_output = os.path.join(output_dir, 'imdb_train.txt')
    test_output = os.path.join(output_dir, 'imdb_test.txt')
    validation_output = os.path.join(output_dir, 'imdb_validation.txt')
    
    # Load the preprocessed dataset
    reviews = load_fasttext_dataset(input_file)
    
    # Split the dataset
    train_set, test_set, validation_set = split_dataset(
        reviews, 
        train_ratio=0.75, 
        test_ratio=0.15, 
        validation_ratio=0.10
    )
    
    # Save each split
    save_dataset(train_set, train_output)
    save_dataset(test_set, test_output)
    save_dataset(validation_set, validation_output)

if __name__ == '__main__':
    main()

# Additional notes:
# 1. Script assumes the following project structure:
#    project_root/
#    ├── scripts/
#    │   └── this_script.py
#    └── Datasets/
#        └── imdb_sentiment_fasttext.txt
# 2. Outputs three files in the Datasets directory:
#    - imdb_train.txt (75% of data)
#    - imdb_test.txt (15% of data)
#    - imdb_validation.txt (10% of data)