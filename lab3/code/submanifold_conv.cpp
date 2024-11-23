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
#include<vector>
using namespace std;

// Configuration constants for convolution parameters
constexpr int batch = 1;              // Number of batches
constexpr int feature_H = 64;         // Height of the input feature map
constexpr int feature_W = 4096;         // Width of the input feature map
constexpr int input_channels = 1;     // Number of input channels
constexpr int output_channels = 64;   // Number of output channels; 64/128/256/512
constexpr int kernel_size = 3;        // Size of the convolution kernel
constexpr int stride = 1;             // Stride of the convolution
constexpr int padding = 0;            // Padding size

// Calculated dimensions for output feature map
constexpr int output_H = feature_H - kernel_size + 1;  // Output height
constexpr int output_W = feature_W - kernel_size + 1;  // Output width

struct coordinate{
    int x, y;
}Coordinate;

// H_in(in repalce of input feature map), filters, and output maps
double input_feature_map[batch][input_channels][feature_H][feature_W];
struct coordinate H_in[batch][input_channels][feature_H * feature_W];
int filter[output_channels][input_channels][kernel_size][kernel_size];
int output_submanifold[batch][output_channels][output_H][output_W];
int output_naive[batch][output_channels][output_H][output_W];

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

// read from preprocessed point cloud data and store in input feature map
void read_pointcloud() {
    ifstream infile("pointcloud_dense.txt");
    if (!infile.is_open()) {
        cerr << "Error: Unable to open file process.txt" << endl;
        return;
    }

    int input_channel_index = 0, batch_index = 0, row_index = 0;
    string line;
    double value;

    while(getline(infile, line)) {
        stringstream ss(line);
        for(int i = 0; i < feature_W; i++){
            ss >> value;
            input_feature_map[batch_index][input_channel_index][row_index][i] = value;
        }

        row_index++;
        if(row_index == feature_H - 1){
            input_channel_index++;
            row_index = 0;
        }
        if(input_channel_index == input_channels){
            batch_index++;
            input_channel_index = 0;
        }
    }
}

// Naive convolution method implementation
void naive_convolution() {
    memset(output_naive, 0, sizeof(output_naive));
    for (int n = 0; n < batch; n++) {
        for (int k = 0; k < output_channels; k++) {
            for (int c = 0; c < input_channels; c++) {
                for (int i = 0; i < output_H; i++) {
                    for (int j = 0; j < output_W; j++) {
                        int sum = 0;
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
void submanifold_convolution(){
    // Step 1: build H_in
    int index = 0;
    for(int n = 0; n < batch; n++){
        for(int c = 0; c < input_channels; c++){
            for(int i = 0; i < feature_H; i++){
                for(int j = 0; j < feature_W; j++){
                    if(input_feature_map[n][c][i][j] != 0.0){
                        H_in[n][c][index].x = i;
                        H_in[n][c][index].y = j;
                        index++;
                    }
                }
            }
        }
    }
    
    // Step 2: build P_out

    // Step 3: build H_out from P_out

    // Step 4: build offset map according to P_out

    // Step 5: build rulebook from H_in, H_out and offset map

    // Step 6: conduct sparse convolution using the rulebook.

}

// Compare results between two output arrays to verify correctness
void test(int output1[batch][output_channels][output_H][output_W], int output2[batch][output_channels][output_H][output_W]) {
    for (int n = 0; n < batch; n++) {
        for (int k = 0; k < output_channels; k++) {
            for (int i = 0; i < output_H; i++) {
                for (int j = 0; j < output_W; j++) {
                    assert(output1[n][k][i][j] == output2[n][k][i][j]);
                }
            }
        }
    }
}


int main(){
    generate_random_kernel();   // Generate input data
    read_pointcloud();          // read from pointcloud.npy and fill it into input_feature_map
    naive_convolution();        // Compute naive convolution

    // Measure time taken by submanifold convolution method
    float avg_time = 0.0f;
    for (int k = 0; k < 32; ++k) {
        auto start_time = get_time();
        submanifold_convolution();
        test(output_naive, output_submanifold);  // Verify result matches naive convolution
        printf("%f\n", get_time() - start_time);
        avg_time += get_time() - start_time;
    }
    std::cout << "Average Time for submanifold convolution: " << (avg_time / 32) << " seconds" << std::endl;
    return 0;
}
