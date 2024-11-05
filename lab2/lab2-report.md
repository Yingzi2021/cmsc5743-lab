# CMSC5743 Lab 2

## Question 1

**Q1** Implement the matrix multiplication using Strassen Algorithm and compare the speed with original matmul() in lab 01. The shape of matrix A is I × K and the shape of matrix B is K × J. The matrix size setting remains the same as lab 01, the value of I, K, J will be fixed at 256, 512 or 1024. 



## Question 2

**Q2** Implement a C++ verison from scratch based on Winograd algorithm and compare the speed with your original im2col implement in lab 01. Please provide analysis on whether or not is your implementation improve the speed performance and why. The Convolution kernel and input size remain the same as lab 01:

- batch: 1 
- height feature: 56 
- width feature: 56 
- in channels: 3 
- out channels: 64 
- kernel size: 3 
- stride: 1 
- padding: 0

参考：

```
https://no5-aaron-wu.github.io/2021/11/16/AI-Algorithm-4-Winograd/
https://github.com/dorthyluu/cs194-winograd/blob/master/winograd.cpp
https://github.com/yester31/Winograd_Conv2d_cpp
```

