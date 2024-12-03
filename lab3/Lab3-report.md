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

## Preprocess

Read the .npy data into a `64 x 4096` matrix

```python
import numpy as np

file_path = "pointcloud.npy"
data = np.load(file_path)
matrix_2d = data.reshape(data.shape[0], -1)
output_file = "pointcloud_dense.txt"
with open(output_file, "w") as f:
    for row in matrix_2d:
        f.write(" ".join(map(str, row)) + "\n")

print(f"Matrix saved to {output_file}")
```

## Sparse Convolution

### Profiling

| Output Channels | Time (s)  |
| --------------- | --------- |
| 64              | 0.0205696 |
| 128             | 0.0397626 |
| 256             | 0.0786029 |
| 512             | 0.14834   |

### Analysis

As the output channel increase, the running time also increase proportionally.