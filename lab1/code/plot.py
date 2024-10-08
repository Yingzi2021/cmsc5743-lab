import matplotlib.pyplot as plt

# # Data from the table
# methods = ['matmul', 'matmul_ikj', 'matmul_AT', 'matmul_BT']
# sizes = [256, 512, 1024]
# times = {
#     'matmul': [0.005820, 0.045579, 0.252103],
#     'matmul_ikj': [0.004952, 0.037697, 0.237074],
#     'matmul_AT': [0.005057, 0.039343, 0.246345],
#     'matmul_BT': [0.005975, 0.042132, 0.260963],
# }

# # Create a figure with 3 subplots in one row
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# # Titles for each subplot
# size_titles = ['Matrix Size: 256', 'Matrix Size: 512', 'Matrix Size: 1024']

# # Plot data for each size
# for i, size in enumerate(sizes):
#     # Get the runtimes for the current size
#     runtimes = [times[method][i] for method in methods]
    
#     # Create a bar plot for the current size
#     axs[i].bar(methods, runtimes, color=['blue', 'orange', 'green', 'red'])
    
#     # Set the title and labels for the subplot
#     axs[i].set_title(size_titles[i])
#     axs[i].set_xlabel('Methods')
#     axs[i].set_ylabel('Runtime (seconds)')
#     axs[i].grid(True)

# # Adjust layout to prevent overlap and save the plot as 'plot.png'
# plt.tight_layout()
# plt.savefig('2.png')

# # Optionally display the plot if needed
# # plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Create the data as seen in the provided image
data = {
    "matrix_size": [256, 512, 1024],
    "O0_matmul": [0.062486, 0.533476, 6.072068],
    "O0_matmul_unrolled": [0.040204, 0.280563, 3.308924],
    "O0_matmul_simd": [0.036931, 0.276264, 3.289734],
    "O1_matmul": [0.020365, 0.165861, 3.026205],
    "O1_matmul_unrolled": [0.017952, 0.137720, 3.159662],
    "O1_matmul_simd": [0.022271, 0.126048, 3.175285],
    "O2_matmul": [0.018086, 0.157981, 3.130562],
    "O2_matmul_unrolled": [0.017981, 0.142533, 3.089540],
    "O2_matmul_simd": [0.022244, 0.128161, 3.281242],
    "O3_matmul": [0.005820, 0.046075, 0.256784],
    "O3_matmul_unrolled": [0.006834, 0.071035, 0.760463],
    "O3_matmul_simd": [0.022283, 0.134812, 3.213914]
}

# Convert the data into a pandas dataframe
df = pd.DataFrame(data)

# Plot four line charts, one for each optimization level (O0, O1, O2, O3)
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# O0 plots
axs[0].plot(df['matrix_size'], df['O0_matmul'], label='matmul', marker='o')
axs[0].plot(df['matrix_size'], df['O0_matmul_unrolled'], label='matmul_unrolled', marker='o')
axs[0].plot(df['matrix_size'], df['O0_matmul_simd'], label='matmul_simd', marker='o')
axs[0].set_title('O0 Optimization')
axs[0].set_xlabel('Matrix Size')
axs[0].set_ylabel('Time(s)')
axs[0].legend()

# O1 plots
axs[1].plot(df['matrix_size'], df['O1_matmul'], label='matmul', marker='o')
axs[1].plot(df['matrix_size'], df['O1_matmul_unrolled'], label='matmul_unrolled', marker='o')
axs[1].plot(df['matrix_size'], df['O1_matmul_simd'], label='matmul_simd', marker='o')
axs[1].set_title('O1 Optimization')
axs[1].set_xlabel('Matrix Size')
axs[1].legend()

# O2 plots
axs[2].plot(df['matrix_size'], df['O2_matmul'], label='matmul', marker='o')
axs[2].plot(df['matrix_size'], df['O2_matmul_unrolled'], label='matmul_unrolled', marker='o')
axs[2].plot(df['matrix_size'], df['O2_matmul_simd'], label='matmul_simd', marker='o')
axs[2].set_title('O2 Optimization')
axs[2].set_xlabel('Matrix Size')
axs[2].legend()

# O3 plots
axs[3].plot(df['matrix_size'], df['O3_matmul'], label='matmul', marker='o')
axs[3].plot(df['matrix_size'], df['O3_matmul_unrolled'], label='matmul_unrolled', marker='o')
axs[3].plot(df['matrix_size'], df['O3_matmul_simd'], label='matmul_simd', marker='o')
axs[3].set_title('O3 Optimization')
axs[3].set_xlabel('Matrix Size')
axs[3].legend()

# Save the figure
plt.tight_layout()
plt.savefig('fig.png')

#plt.show()
