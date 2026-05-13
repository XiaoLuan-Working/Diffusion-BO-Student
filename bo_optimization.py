import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Integer
import numpy as np
import logging
from evaluate import evaluate_generation

logger = logging.getLogger(__name__)

class BayesianOptimizer:
    """
    Bayesian Optimizer for searching the optimal number of inference steps.
    """
    def __init__(self, model, prompt, seed=42):
        self.model = model
        self.prompt = prompt
        self.seed = seed
        self.results_history = []
        
    def objective_function(self, params):
        steps = params[0]
        logger.info(f"[BO Search] Testing steps: {steps}")
        
        image, inference_time = self.model.generate_image(
            prompt=self.prompt,
            steps=steps,
            seed=self.seed
        )
        
        eval_metrics = evaluate_generation(image, inference_time)
        score = eval_metrics["score"]
        
        logger.info(f"  -> Time: {inference_time:.2f}s, Object Sharpness: {eval_metrics['sharpness']:.2f}")
        logger.info(f"  -> Score: {score:.2f}")
        
        self.results_history.append((steps, score, eval_metrics['sharpness'], inference_time))
        
        return -score

    def run_optimization(self, n_calls=15):
        """
        Execute the Bayesian Optimization loop.
        """
        logger.info("="*50)
        logger.info("Starting Bayesian Optimization")
        logger.info("="*50)
        
        search_space = [Integer(10, 200, name='steps')]
        
        res = gp_minimize(
            func=self.objective_function,
            dimensions=search_space,
            n_calls=n_calls,
            n_initial_points=min(5, n_calls),
            random_state=self.seed,
            verbose=False
        )
        
        best_steps = res.x[0]
        best_score = -res.fun
        
        logger.info("="*50)
        logger.info(f"Optimization Finished! Best Sampling Steps: {best_steps} (Score: {best_score:.2f})")
        logger.info("="*50)
        
        return best_steps, best_score, self.results_history

    def plot_optimization_curve(self, save_path="bo_curve.png"):
        """
        Plot and save the BO optimization progress curve.
        """
        steps = [h[0] for h in self.results_history]
        scores = [h[1] for h in self.results_history]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(steps, scores, color='blue', label='Evaluated Points')
        
        sorted_indices = np.argsort(steps)
        plt.plot(np.array(steps)[sorted_indices], np.array(scores)[sorted_indices], 'r--', alpha=0.5)
        
        plt.xlabel("Sampling Steps")
        plt.ylabel("Objective Score")
        plt.title("Bayesian Optimization: Steps vs Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        logger.info(f"Optimization curve saved to {save_path}")
        plt.close()
