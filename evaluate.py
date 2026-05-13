import cv2
import numpy as np
from PIL import Image

def calculate_sharpness(image: Image.Image) -> float:
    """
    Calculate the Laplacian sharpness of an image.
    
    Args:
        image (PIL.Image.Image): The generated image to evaluate.
        
    Returns:
        float: The variance of the Laplacian map.
    """
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(cv_img, cv2.CV_64F).var()
    return float(variance)

def evaluate_generation(image: Image.Image, inference_time: float, lambda_penalty: float = 10.0) -> dict:
    """
    Evaluate generation quality using a composite score.
    Higher score indicates better quality vs time trade-off.
    
    Args:
        image (PIL.Image.Image): The generated image.
        inference_time (float): Time taken for generation.
        lambda_penalty (float): Weight for time penalty.
        
    Returns:
        dict: Evaluation metrics including sharpness, time, and score.
    """
    sharpness = calculate_sharpness(image)
    score = sharpness - lambda_penalty * inference_time
    
    return {
        "sharpness": sharpness,
        "inference_time": inference_time,
        "score": score
    }
