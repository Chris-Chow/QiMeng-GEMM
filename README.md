# QiMeng-GEMM: Automatically Generating High-Performance Matrix Multiplication Code by Exploiting Large Language Models

<img src="static/overview.png" style="zoom:50%;" /> 

## Quick Start
### CUDA
Use this to call the GEMM cuda kernel
```cpp
...
cuda_gemm(M, N, K, alpha, d_A, d_B, beta, d_C);  
...
```

## Paper
This paper has beed accepted by AAAI2025 poster.

Please cite this paper if you use this code.

```
@article{
  title={QiMeng-GEMM: Automatically Generating High-Performance Matrix Multiplication Code by Exploiting Large Language Models}, 
  volume={39}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/34461}, 
  DOI={10.1609/aaai.v39i21.34461}, 
  number={21},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Zhou, Qirui and Wen, Yuanbo and Chen, Ruizhi and Gao, Ke and Xiong, Weiqiang and Li, Ling and Guo, Qi and Wu, Yanjun and Chen, Yunji}, 
  year={2025}, 
  month={Apr.}, 
  pages={22982-22990}
}
    
```
