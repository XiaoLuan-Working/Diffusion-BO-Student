<img width="1674" height="77" alt="image" src="https://github.com/user-attachments/assets/6d6c5cbe-5824-4eae-bff9-8ad042653289" /># Diffusion Sampling Optimization Lab (Student Template)

This lab provides an implementation framework to demonstrate and optimize the denoising iterations ("sampling steps") within Diffusion Models using Bayesian Optimization (BO). 

**This is a student assignment template. Some core integration steps have been left as TODOs.**

## 环境要求 (Environment Setup)

Please install dependencies using python 3.8+:
```bash
pip install -r requirements.txt
```

*(GPU is auto-detected. Run `pip install accelerate` if `low_cpu_mem_usage` issues occur during model loading.)*
如果没有GPU记得把参数调小，要不然电脑运行会很慢
## 实验任务 (Lab Tasks)

本实验分为两个递进的任务阶段，请编辑 `main.py` 并在终端运行来观察结果。

### Task 1: 手动探索去噪过程 (Manual Exploration)
在 `main.py` 中，程序已经默认提供了一段可运行的代码，它会以固定的步数（默认 `test_steps = 20`）生成一张图片。
- **你的任务**：尝试修改 `main.py` 中的 `test_steps` 变量（比如改为 10, 50, 100），重新运行并观察生成图像保存在 `results/` 目录下质量的变化，以及终端显示的推理时间 (Inference Time)。

### Task 2: 接入贝叶斯优化 (Integrate Bayesian Optimization)
既然步数太少会导致图像崩坏，步数太多会导致时间无限拉长，我们需要通过算法自动寻找**效率与质量的最佳平衡点**。
在项目中已经为您提供写好的 `bo_optimization.py` 优化器。
- **你的任务**：
  1. 在 `main.py` 顶部导入 `BayesianOptimizer`。
  2. 在 `main.py` 底部的 TODO 区域，实例化该优化器。
  3. 调用其搜索接口自动寻找最佳步数。
  4. 使用寻找到的最佳步数，用模型重新生成一张最终极的优化图像并保存。

## 运行方式 (Usage)

完成代码补充后，通过以下命令运行完整测试（国内网络推荐加上镜像环境变量）：

```bash
export HF_ENDPOINT=https://hf-mirror.com && python main.py --out_dir results
```

You can customize testing parameters:
- `--prompt`: Description of the image.
- `--seed`: Deterministic generation seed.

## 核心算法释义 (Implementation Details & Engineering Principles)

### 1. 为什么优化 Sampling Steps (去噪步数)？
原始扩散模型（例如 DDPM）常需要数百到上千次推理迭代才能将纯高斯噪声完全去噪至干净画幅。
我们通过调节 `steps`，目标是**不改动 U-Net 结构的前提下**寻找可以生成极高质量图片的最少迭代数字。

### 2. 什么是客观质量评价标准（Evaluate.py 中的 Sharpness 模型）
工业界常采用 FID 或 CLIP Score，因受限于教学环境复杂配置，本项目选用了**拉普拉斯方差 (Laplacian Sharpness)** 代替。通过把生成的 PIL 图像转换成灰度 cv2 矩阵，再计算拉普拉斯层方差。方差值可以有效衡量图像的**边缘锐利程度和细节纹理量**。方差越大视作生成质量越好。

### 3. 为什么引入贝叶斯优化（Bayesian Optimization）？
目标函数（Object Function）考虑了多目标权衡：
> **Score = Sharpness (质量) - λ * Inference_Time (耗时惩罚)**
由于一次完整的 Diffusion 前向推进极其慢，Grid Search 或 Random Search 代价太高，贝叶斯优化可以在非常少的目标函数评估（大概十几轮之内）的代价下，定位最高效的最优采样步数值。
### (可选) 本地模型权重导入加载方式
如果遇到 HuggingFace 下载卡顿甚至断网的情况，可以提前将离线模型下载好，然后在代码中进行本地加载。
- 国内镜像仓库地址:https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5/files
- v1-5-pruned-emaonly.safetensors（这个文件大概4.27GB）

#### 如何使用离线文件？
1. 从上述地址将该仓库里的所有文件克隆/下载到你的电脑中（或通过学校机房内网传包），一般需要下载所有 `.json` 配置文件和 `weight.bin`/`safetensors` 文件，放在一个你自己新建的 `stable-diffusion-v1-5` 文件夹里。
2. 打开 `diffusion_model.py` 文件，把原来的在线名字 `"runwayml/stable-diffusion-v1-5"` 替换为你刚保存的那个文件夹的绝对路径。
   例如：`model_id = "/绝对路径/到你的/stable-diffusion-v1-5/"`


