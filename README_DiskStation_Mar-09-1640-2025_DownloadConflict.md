## [When Random Is Not Enough: Exploring Automatic Truncated Optimization for Dataset Distillation](https://github.com/GH209515338/AT-BPTT/blob/main/README.md)
---
#### [Project Page]() | [Paper]() 
---
### Abstract
The growing demand for efficient deep learning has positioned dataset distillation as a pivotal technique for compressing training dataset while preserving model performance. However, existing inner-loop optimization methods in bilevel dataset distillation suffer from suboptimal gradient dynamics due to uniform truncation strategies across training stages. This paper identifies that neural networks exhibit distinct learning patterns during early, middle, and late training stages, rendering rigid truncation approaches ineffective. To address this challenge, we propose the Automatic Truncated Backpropagation Through Time (AT-BPTT), a novel framework that flexibly adapts truncation positions and window sizes to align with intrinsic gradient dynamics. AT-BPTT integrates three key innovations: a probabilistic truncation mechanism for stage-aware timestep selection, an adaptive window sizing strategy driven by gradient variation analysis, and a threshold-guided transition protocol for stable stage switching. Extensive experiments on CIFAR-10, CIFAR-100, Tiny-ImageNet, and ImageNet-1 K demonstrate that AT-BPTT achieves state-of-the-art performance, outperforming the baseline method by an average of 2.4% in accuracy.  
### Introduction
Our contributions are summarized as follows: 
- We establishes an intrinsic connection between the gradient dynamics governing DNN training process and the BPTT framework, thereby highlighting the critical need for automatic truncation optimization.
- We propose an AT-BPTT framework, which seamlessly integrates Dynamic Truncation Position, Adaptive Window Size, and Threshold-guided Stage Transition to optimize the inner-loop training process.
- Extensive experiments illustrate that AT-BPTT achieves state-of-the-art performance on CIFAR-10 , CIFAR-100 , Tiny-ImageNet and ImageNet-1K  outperforming RaT-BPTT by an average of 2.4\%.

![Method](python_at_bptt/Figure/method.png)
### Performance
![Performance](Figure/performance.png)
Under various IPC settings across multiple standard datasets, our method holds a leading position.
### Visualization of Synthetic Images
**CIFAR-10**:
![[10.png|700]]
**CIFAR-100**：
![[100.png|700]]
### Getting Started
*The code is built upon [RaT-BPTT](https://github.com/fengyzpku/Simple_Dataset_Distillation). If you utilize the code, please cite their paper.*
To get startd with AT-BPTT, as follows:
1. Clone the repository
```python
git clone https://github.com/GH209515338/AT-BPTT.git
```
2. Create environment and install dependencies
```python
conda env create -f environment.yml
conda activate atbptt
```
3. Example 









