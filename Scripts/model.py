import os
import fasttext
import argparse
import logging
from typing import Optional

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

def train_fasttext_model(
    input_path: str, 
    output_path: str, 
    model_name: str = 'sentiment_classifier',
    epochs: int = 10, 
    lr: float = 1.0, 
    wordNgrams: int = 2,
    dim: int = 100,
    minCount: int = 10,
    loss: str = 'softmax'
) -> Optional[fasttext.FastText._FastText]:
    logger = setup_logging()
    
    try:
        os.makedirs(output_path, exist_ok=True)
        
        model_file_path = os.path.join(output_path, f'{model_name}.bin')
        
        logger.info(f"Starting model training with parameters:")
        logger.info(f"Input file: {input_path}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Learning Rate: {lr}")
        logger.info(f"Word N-grams: {wordNgrams}")
        
        model = fasttext.train_supervised(
            input=input_path,
            lr=lr,
            epoch=epochs,
            wordNgrams=wordNgrams,
            dim=dim,
            minCount=minCount,
            loss=loss,
            verbose=1,
        )
        
        save_model(model, model_file_path)
        logger.info(f"Model saved to: {model_file_path}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None

def save_model(
    model: fasttext.FastText._FastText, 
    file_path: str
) -> bool:
    """
    Save a FastText model to a specified file path.
    
    Args:
        model (fasttext.FastText._FastText): The trained FastText model
        file_path (str): Full path where the model should be saved
    
    Returns:
        bool: True if model was saved successfully, False otherwise
    """
    logger = setup_logging()
    
    try:
        model.save_model(file_path)
        logger.info(f"Model successfully saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def load_model(
    file_path: str
) -> Optional[fasttext.FastText._FastText]:
    """
    Load a FastText model from a specified file path.
    
    Args:
        file_path (str): Full path to the model file
    
    Returns:
        Optional[fasttext.FastText._FastText]: Loaded model or None if loading fails
    """
    logger = setup_logging()
    
    try:
        model = fasttext.load_model(file_path)
        logger.info(f"Model successfully loaded from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def evaluate_model(
    model: fasttext.FastText._FastText, 
    test_path: str
) -> dict:
    
    logger = setup_logging()
    
    try:
        results = model.test(test_path)
        
        logger.info("Model Evaluation Results:")
        logger.info(f"Number of samples: {results[0]}")
        logger.info(f"Precision @ 1: {results[1]}")
        logger.info(f"Recall @ 1: {results[2]}")
        
        return {
            'samples': results[0],
            'precision': results[1],
            'recall': results[2]
        }
    
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        return {}

def predict_sentiment(
    model: fasttext.FastText._FastText, 
    text: str
) -> str:
    
    text = text.lower().strip()
    
    prediction = model.predict(text)
    
    return prediction[0][0].replace('__label__', '')

def main():
    project_root = get_project_root()
    
    datasets_dir = os.path.join(project_root, 'Datasets')
    models_dir = os.path.join(project_root, 'Models')
    
    train_path = os.path.join(datasets_dir, 'imdb_train.txt')
    test_path = os.path.join(datasets_dir, 'imdb_test.txt')
    validation_path = os.path.join(datasets_dir, 'imdb_validation.txt')
    
    parser = argparse.ArgumentParser(description='FastText Sentiment Classification')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--wordngrams', type=int, default=2, help='Word n-grams')
    
    args = parser.parse_args()
    
    model = train_fasttext_model(
        input_path=train_path,
        output_path=models_dir,
        epochs=args.epochs,
        lr=args.lr,
        wordNgrams=args.wordngrams
    )
    
    if model:
        evaluation_results = evaluate_model(model, test_path)
        
        example_texts = [
            "This movie was absolutely fantastic and brilliantly acted!",
            "I was completely bored and disappointed by this terrible film."
        ]
        
        print("\nExample Predictions:")
        for text in example_texts:
            sentiment = predict_sentiment(model, text)
            print(f"Text: {text}")
            print(f"Predicted Sentiment: {sentiment}\n")

if __name__ == '__main__':
    main()