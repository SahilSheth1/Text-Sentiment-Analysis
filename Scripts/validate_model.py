import os
import logging
from model import get_project_root, load_model

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def validate_model(model_path, validation_path):
    logger = setup_logging()
    
    try:
        model = load_model(model_path)
        if not model:
            logger.error("Failed to load the model.")
            return {}
        
        results = model.test(validation_path)
        
        # Detailed analysis
        total_samples = results[0]
        precision = results[1]
        recall = results[2]
        
        logger.info("Validation Results:")
        logger.info(f"Total Samples: {total_samples}")
        logger.info(f"Precision @ 1: {precision}")
        logger.info(f"Recall @ 1: {recall}")
        
        correct_predictions = 0
        total_predictions = 0
        
        with open(validation_path, 'r', encoding='utf-8') as f:
            prediction_details = []
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    label, text = line.strip().split(' ', 1)
                except ValueError:
                    logger.warning(f"Skipping invalid line: {line}")
                    continue
                
                try:
                    prediction = model.predict(text)[0][0].replace('__label__', '')
                    
                    is_correct = prediction == label
                    correct_predictions += 1 if is_correct else 0
                    total_predictions += 1
                    
                    prediction_details.append({
                        'text': text,
                        'true_label': label,
                        'predicted_label': prediction,
                        'is_correct': is_correct
                    })
                except Exception as e:
                    logger.error(f"Error predicting sentiment: {e}")
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        metrics = {
            'total_samples': total_samples,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'prediction_details': prediction_details
        }
        
        misclassified = [
            detail for detail in prediction_details if not detail['is_correct']
        ]
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Total Predictions: {total_predictions}")
        logger.info(f"Correct Predictions: {correct_predictions}")
        
        print("\nSample Misclassified Predictions:")
        for detail in misclassified[:5]: 
            print(f"Text: {detail['text']}")
            print(f"True Label: {detail['true_label']}")
            print(f"Predicted Label: {detail['predicted_label']}\n")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error during model validation: {e}")
        return {}

def main():
    project_root = get_project_root()
    models_dir = os.path.join(project_root, 'Datasets')
    models_path = os.path.join(project_root, 'Models')
    
    model_path = os.path.join(models_path, 'sentiment_classifier.bin')
    validation_path = os.path.join(models_dir, 'imdb_validation.txt')
    
    validation_results = validate_model(model_path, validation_path)
    
    if validation_results:
        import json
        results_path = os.path.join(models_path, 'validation_results.json')
        with open(results_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"Detailed validation results saved to {results_path}")

if __name__ == '__main__':
    main()