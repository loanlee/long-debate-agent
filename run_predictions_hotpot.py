import argparse
import jsonlines
import os
import json
from tqdm import tqdm
from typing import Callable, List, Dict, Any

from gemini_model import setup_gemini, gemini_predict

def load_dataset(dataset_path: str, max_samples: int = None) -> List[Dict]:
    """Load the dataset from a JSONL file.
    
    Args:
        dataset_path: Path to the JSONL dataset file
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        List of dictionaries containing the dataset examples
    """
    data = []
    with jsonlines.open(dataset_path) as reader:
        for item in reader:
            data.append(item)
            if max_samples and len(data) >= max_samples:
                break
    return data

def prepare_prompt(item: Dict) -> str:
    """Prepare the prompt for a single example.
    
    Args:
        item: Dictionary containing 'context' and 'question' fields
        
    Returns:
        Formatted prompt string
    """
    context = item['context']
    question = item['input']
    prompt = f"""Context: {context}

    Question: {question}

    Answer: """
    return prompt

def run_predictions_hotpot(
    dataset_path: str,
    predict_fn: Callable[[str], str],
    max_samples: int = None,
    output_path: str = "predictions.json"
) -> List[Dict[str, Any]]:
    """
    Run predictions on the dataset and save raw predictions.
    
    Args:
        dataset_path: Path to the JSONL dataset file
        predict_fn: Function that takes a formatted prompt and returns a prediction
        max_samples: Maximum number of samples to evaluate (None for all)
        output_path: Path to save the predictions JSON file
    
    Returns:
        List of prediction dictionaries
    """
    # Use our existing load_dataset function
    data = load_dataset(dataset_path, max_samples)
    
    # Check for existing output and get completed IDs
    completed_ids = set()
    predictions = []
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                pred = json.loads(line)
                completed_ids.add(pred["id"])
                predictions.append(pred)
    
    # Open file in append mode
    with open(output_path, "a") as f:
        for idx, item in enumerate(tqdm(data, desc="Running predictions")):
            if item["id"] in completed_ids:
                continue  # Skip already completed
            
            # Use our existing prepare_prompt function
            prompt = prepare_prompt(item)
            
            # Get prediction from model
            prediction = predict_fn(prompt)
            
            # Format result
            result = {
                "id": idx,
                "context_length": item['context_length'],
                "depth_percent1": item['depth_percent1'],
                "depth_percent2": item['depth_percent2'],
                "input": item['input'],
                "dataset": item['dataset'],
                "ground_truth": item['answers'],
                "prediction": prediction
            }
            f.write(json.dumps(result) + "\n")
            f.flush()  # Ensure it's written to disk
            predictions.append(result)
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Run Gemini predictions on Hotpot-style dataset.")
    parser.add_argument('--dataset_path', type=str, default='NeedleInAHaystack-PLUS/needle_plus_hotpotqa.jsonl', help='Path to the input JSONL dataset')
    parser.add_argument('--output_path', type=str, default='gemini_predictions_hotpotqa.json', help='Path to save predictions')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to predict')
    args = parser.parse_args()

    setup_gemini()
    predictions = run_predictions_hotpot(
        dataset_path=args.dataset_path,
        predict_fn=gemini_predict,
        max_samples=args.max_samples,
        output_path=args.output_path
    )
    print(f"Saved {len(predictions)} predictions to {args.output_path}")
    
if __name__ == "__main__":
    main()
