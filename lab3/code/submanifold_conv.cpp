#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <unordered_map>
using namespace std;

// Configuration constants for convolution parameters
constexpr int batch = 1;              // Number of batches
constexpr int feature_H = 64;         // Height of the input feature map
constexpr int feature_W = 4096;       // Width of the input feature map
constexpr int input_channels = 1;     // Number of input channels
constexpr int output_channels = 512;   // Number of output channels
constexpr int kernel_size = 3;        // Size of the convolution kernel
constexpr int stride = 1;             // Stride of the convolution
constexpr int padding = 0;            // Padding size

// Calculated dimensions for output feature map
constexpr int output_H = feature_H - kernel_size + 1;  // Output height
constexpr int output_W = feature_W - kernel_size + 1;  // Output width
constexpr int kernel_radius = kernel_size / 2;         // Radius of the kernel

struct coordinate {
    int x, y;
};

double input_feature_map[batch][input_channels][feature_H][feature_W];
bool input_nonzero[batch][input_channels][feature_H][feature_W];
coordinate H_in[batch][input_channels][feature_H * feature_W];
int H_in_count[batch][input_channels];
int filter[output_channels][input_channels][kernel_size][kernel_size];
double output_submanifold[batch][output_channels][output_H][output_W];
double output_naive[batch][output_channels][output_H][output_W];

int offsets[9][2] = {
        {-1, -1}, {-1, 0}, {-1, +1},
        {0, -1},  {0, 0},  {0, +1},
        {+1, -1}, {+1, 0}, {+1, +1}
};

// Get current time in seconds (for benchmarking)
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// Generate random values for input feature maps and filters
void generate_random_kernel() {
    srand(static_cast<unsigned>(time(0)));
    for (int k = 0; k < output_channels; k++) {
        for (int c = 0; c < input_channels; c++) {
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    filter[k][c][i][j] = rand() % 10;
                }
            }
        }
    }
}

// Read from preprocessed point cloud data and store in input feature map
void read_pointcloud() {
    ifstream infile("pointcloud_dense.txt");
    if (!infile.is_open()) {
        cerr << "Error: Unable to open file pointcloud_dense.txt" << endl;
        return;
    }

    int batch_index = 0, input_channel_index = 0, row_index = 0;
    string line;
    double value;

    while (getline(infile, line)) {
        stringstream ss(line);
        for (int i = 0; i < feature_W; i++) {
            ss >> value;
            input_feature_map[batch_index][input_channel_index][row_index][i] = value;
        }
        row_index++;
        if (row_index == feature_H) {
            input_channel_index++;
            row_index = 0;
        }
        if (input_channel_index == input_channels) {
            batch_index++;
            input_channel_index = 0;
        }
    }

    infile.close();
}

// Naive convolution method implementation
void naive_convolution() {
    memset(output_naive, 0, sizeof(output_naive));
    for (int n = 0; n < batch; n++) {
        for (int k = 0; k < output_channels; k++) {
            for (int c = 0; c < input_channels; c++) {
                for (int i = 0; i < output_H; i++) {
                    for (int j = 0; j < output_W; j++) {
                        double sum = 0.0;
                        for (int p = 0; p < kernel_size; p++) {
                            for (int q = 0; q < kernel_size; q++) {
                                sum += input_feature_map[n][c][i + p][j + q] * filter[k][c][p][q];
                            }
                        }
                        output_naive[n][k][i][j] += sum;
                    }
                }
            }
        }
    }
}

// Implement submanifold convolution
void submanifold_convolution() {
    memset(output_submanifold, 0, sizeof(output_submanifold));

    // Build H_in and input_nonzero
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < input_channels; c++) {
            int index = 0;
            for (int i = kernel_radius; i < feature_H - kernel_radius; i++) {
                for (int j = kernel_radius; j < feature_W - kernel_radius; j++) {
                    input_nonzero[n][c][i][j] = (input_feature_map[n][c][i][j] != 0.0);
                    if (input_nonzero[n][c][i][j]) {
                        H_in[n][c][index].x = i;
                        H_in[n][c][index].y = j;
                        index++;
                    }
                }
            }
            H_in_count[n][c] = index;
        }
    }

    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < input_channels; c++) {
            for (int idx = 0; idx < H_in_count[n][c]; idx++) {
                int i = H_in[n][c][idx].x;
                int j = H_in[n][c][idx].y;
                int output_i = i - kernel_radius;
                int output_j = j - kernel_radius;

                if (output_i >= 0 && output_i < output_H && output_j >= 0 && output_j < output_W) {
                    for (int k = 0; k < output_channels; k++) {
                        double sum = 0.0;
                        for (int c_prime = 0; c_prime < input_channels; c_prime++) {
                            for (int o = 0; o < 9; o++) {
                                int p = offsets[o][0];
                                int q = offsets[o][1];
                                int ni = i + p;
                                int nj = j + q;
                                if (ni >= 0 && ni < feature_H && nj >= 0 && nj < feature_W) {
                                    if (input_nonzero[n][c_prime][ni][nj]) {
                                        sum += input_feature_map[n][c_prime][ni][nj] * filter[k][c_prime][p + kernel_radius][q + kernel_radius];
                                    }
                                }
                            }
                        }
                        output_submanifold[n][k][output_i][output_j] += sum;
                    }
                }
            }
        }
    }
}

// Compare results between two output arrays to verify correctness
void test(double output1[batch][output_channels][output_H][output_W], double output2[batch][output_channels][output_H][output_W]) {
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < input_channels; c++) {
            for (int idx = 0; idx < H_in_count[n][c]; idx++) {
                int i = H_in[n][c][idx].x;
                int j = H_in[n][c][idx].y;
                int output_i = i - kernel_radius;
                int output_j = j - kernel_radius;
                if (output_i >= 0 && output_i < output_H && output_j >= 0 && output_j < output_W) {
                    for (int k = 0; k < output_channels; k++) {
                        assert(abs(output1[n][k][output_i][output_j] - output2[n][k][output_i][output_j]) < 1e-5);
                    }
                }
            }
        }
    }
}

int main() {
    generate_random_kernel();   // Generate random kernels
    read_pointcloud();          // Read from pointcloud_dense.txt and fill it into input_feature_map
    naive_convolution();        // Compute naive convolution

    // Measure time taken by submanifold convolution method
    double avg_time = 0.0;
    for (int k = 0; k < 32; ++k) {
        double start_time = get_time();
        submanifold_convolution();
        test(output_naive, output_submanifold);  // Verify result matches naive convolution
        double elapsed = get_time() - start_time;
        printf("Iteration %d: %f seconds\n", k + 1, elapsed);
        avg_time += elapsed;
    }
    std::cout << "Average Time for submanifold convolution: " << (avg_time / 32) << " seconds" << std::endl;
    return 0;
}
