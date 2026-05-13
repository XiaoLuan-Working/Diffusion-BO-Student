import os
import argparse
import logging
from diffusion_model import DiffusionModel
# from bo_optimization import ... (待学生自己导入和接入)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Diffusion Sampling Optimization Lab")
    parser.add_argument("--prompt", type=str, default="A beautiful photo of an astronaut riding a horse on mars, high detail, 4k", help="Prompt for image generation")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save experimental results")
    args = parser.parse_args()
    
    logger.info("Welcome to Diffusion Sampling Optimization Lab!")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 初始化扩散模型
    model = DiffusionModel()
    

    test_steps = 30  # 可以修改该数值观察结果
    logger.info(f"[Manual Test] Generating with {test_steps} steps...")
    
    img, inf_time = model.generate_image(args.prompt, test_steps, args.seed)
    
    file_path = os.path.join(args.out_dir, f"manual_step_{test_steps}.png")
    img.save(file_path)
    logger.info(f"Saved: {file_path} | Time: {inf_time:.2f}s")
    
    # ---------------------------------------------------------
    # 【第二部分：学生任务 - 接入 Bayesian Optimization】
    # ---------------------------------------------------------
    logger.info("="*50)
    logger.info("TODO: 请在此接入 bo_optimization.py 里的优化器。")
    logger.info("通过贝叶斯优化，自动寻找既快又清晰的最佳去噪步数！")
    logger.info("="*50)
    
    # TODO: 实例化 BayesianOptimizer
    # TODO: 调用 run_optimization() 自动搜索最佳步数
    # TODO: 使用 BO 搜索到的最佳步数，重新生成一张最终图像并保存

if __name__ == "__main__":
    main()
