# QiMeng-GEMM: Automatically Generating High-Performance Matrix Multiplication Code by Exploiting Large Language Models
QiMeng-GEMM is an innovative approach to automatically generate high-performance matrix multiplication (GEMM) code using LLMs. This codebase provides a comprehensive solution for efficiently computing matrix multiplications, leveraging the power of LLMs to generate optimized code based on user inputs.

## Quick Start
### CUDA

To test the performance of cuda code, you can run this:
```bash
cd code/cuda
nvcc -o test main.cpp kernel_128_128.cu kernel_32_64.cu kernel_64_128.cu kernel_64_64.cu kernel.cu  -lcublas
./test [M] [N] [K]
```

You can use this to call the GEMM cuda kernel
```cpp
cuda_gemm(M, N, K, alpha, d_A, d_B, beta, d_C);  
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
