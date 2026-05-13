import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def visualize_samples(images_dict, save_path="comparison.png"):
    """
    Visualize standard step variations side-by-side.
    
    Args:
        images_dict (dict): Dictionary mapping steps (int) to PIL.Image.
        save_path (str): Filepath for saving the visualization.
    """
    steps_list = list(images_dict.keys())
    n = len(steps_list)
    
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
        
    for ax, steps in zip(axes, steps_list):
        ax.imshow(images_dict[steps])
        ax.axis('off')
        ax.set_title(f"Steps: {steps}")
        
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Visualization saved to {save_path}")
    plt.close()
