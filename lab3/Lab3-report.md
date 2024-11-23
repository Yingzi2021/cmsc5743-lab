# CMSC5743 Lab 3

## Question 1

**Q1** Read the given “pointcloud.npy” data, which is a 64×4096 matrix. Implement a C++ version of sparse convolution and record the inference time with different out channel numbers. Convolution parameters are given: 

- batch: 1 
- height feature: 64 
- width feature: 4096 
- in channels: 1 
- out channels: 64/128/256/512 
- kernel size: 3 
- stride: 1 
- padding: 0 

Analyze the relationship between inference time and output channel