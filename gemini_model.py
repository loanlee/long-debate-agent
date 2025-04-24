import os
import random
from dotenv import load_dotenv
import google.generativeai as genai
from time import sleep

def setup_gemini() -> None:
    """Initialize the Gemini model with API key from environment."""
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)

def gemini_predict(prompt: str, max_retries: int = 5) -> str:
    """
    Generate predictions using Gemini model with exponential backoff.
    
    Args:
        prompt: The formatted prompt string
        max_retries: Maximum number of retries on API error
        
    Returns:
        The model's prediction as a string
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                return "Error: Response blocked"
                
            if response.text:
                return response.text.strip()
            
            return "Error: Empty response"
            
        except Exception as e:
            # Calculate exponential backoff time with some randomness (jitter)
            backoff_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Error (attempt {attempt+1}/{max_retries}): {str(e)}")
            print(f"Retrying in {backoff_time:.2f} seconds...")
            sleep(backoff_time)
    
    return "Error: Maximum retries exceeded"
