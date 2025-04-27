You are an expert in CUDA C programming. You can help me to generate the corresponding CUDA C code as per my instruction requirements. What you should do is generate a CUDA C program to realize general matrix multiply, thus the matrix multiplication of matrix A of dimension M×K and matrix B of dimension K×N is realized and stored in matrix C. The The final realization is C = αAB + βC. The parameters needed in function are provided in the definition of the function. The kernel function should be like:

```cpp\ntemplate<...>\n__global__ void gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){\n\t\t//matrix multiplication part\n}\n```

Here are some concrete code requirements.

-  Just generate the code of kernel function;
- Try not to use "if & else" statements to improve code efficiency.