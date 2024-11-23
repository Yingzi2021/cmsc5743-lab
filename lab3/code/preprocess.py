import numpy as np

file_path = "pointcloud.npy"
data = np.load(file_path)
matrix_2d = data.reshape(data.shape[0], -1)
output_file = "pointcloud_dense.txt"
with open(output_file, "w") as f:
    for row in matrix_2d:
        f.write(" ".join(map(str, row)) + "\n")

print(f"Matrix saved to {output_file}")
