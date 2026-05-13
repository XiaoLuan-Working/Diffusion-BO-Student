import torch
from diffusers import StableDiffusionPipeline
import time
import logging

logger = logging.getLogger(__name__)

class DiffusionModel:
    """
    Wrapper for the Diffusion Model.
    Provides image generation with configurable inference steps.
    """
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Diffusion Model onto {self.device}...")
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            requires_safety_checker=False,
            safety_checker=None
        )
        self.pipe = self.pipe.to(self.device)
        logger.info("Model loaded successfully.")

    def generate_image(self, prompt, steps, seed, save_intermediate=False):
        """
        Generates an image based on a text prompt.
        
        Args:
            prompt (str): Text prompt for generation.
            steps (int): Number of inference/denoising steps.
            seed (int): Random seed for reproducibility.
            save_intermediate (bool): Whether to save intermediate latents (placeholder).
            
        Returns:
            final_image (PIL.Image.Image): The generated image.
            inference_time (float): Time taken for inference.
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        intermediate_images = []
        
        start_time = time.time()
        
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            generator=generator,
        )
        
        inference_time = time.time() - start_time
        final_image = result.images[0]
        
        if save_intermediate:
            return final_image, intermediate_images, inference_time
        
        return final_image, inference_time
