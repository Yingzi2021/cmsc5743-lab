#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>

constexpr int batch = 2;
constexpr int feature_H = 56;
constexpr int feature_W = 56;
constexpr int input_channels = 3; 
constexpr int output_channels = 3; 
constexpr int kernel_size = 3;
constexpr int stride = 1;
constexpr int padding = 0;

constexpr int output_H = (feature_H - kernel_size + 2 * padding) / stride + 1;
constexpr int output_W = (feature_W - kernel_size + 2 * padding) / stride + 1;

int input_feature_map[batch][input_channels][feature_H][feature_W];  
int filter[output_channels][input_channels][kernel_size][kernel_size];
int output_feature_map[batch][output_channels][output_H][output_W];
int output_naive[batch][output_channels][output_H][output_W];

int im_col[batch][input_channels * kernel_size * kernel_size][output_H * output_W]; // convert input feature map to cols
int filter_col[output_channels][input_channels * kernel_size * kernel_size]; // convert filter to cols

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// The input2col function that will convert the input feature map to column format
void input2col() {
    for (int n = 0; n < batch; ++n) {
        for (int h_out = 0; h_out < output_H; ++h_out) {
            for (int w_out = 0; w_out < output_W; ++w_out) {
                int col_index = h_out * output_W + w_out;  // This is the column index in im_col
                for (int c = 0; c < input_channels; ++c) {
                    int channel_offset = c * kernel_size * kernel_size;
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int h_in = h_out * stride + kh - padding;
                            int w_in = w_out * stride + kw - padding;
                            if (h_in >= 0 && h_in < feature_H && w_in >= 0 && w_in < feature_W) {
                                // Store the value from input feature map to the corresponding position in im_col
                                im_col[n][channel_offset + kh * kernel_size + kw][col_index] = 
                                    input_feature_map[n][c][h_in][w_in];
                            } else {
                                // Padding: if the input location is outside the bounds, set it to 0
                                im_col[n][channel_offset + kh * kernel_size + kw][col_index] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}

// The filter2col function that will convert the filters to column format
void filter2col() {
    for (int oc = 0; oc < output_channels; ++oc) {
        for (int ic = 0; ic < input_channels; ++ic) {
            int col_index = ic * kernel_size * kernel_size;
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    filter_col[oc][col_index + kh * kernel_size + kw] = 
                        filter[oc][ic][kh][kw];
                }
            }
        }
    }
}

// Matrix multiplication to calculate the output feature map
void matmul() {
    // Initialize output_feature_map to 0
    memset(output_feature_map, 0, sizeof(output_feature_map));

    // Perform matrix multiplication
    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < output_channels; ++oc) {
            for (int i = 0; i < output_H * output_W; ++i) { // Iterate over output feature map positions
                for (int j = 0; j < input_channels * kernel_size * kernel_size; ++j) { // Iterate over filter_col and im_col
                    output_feature_map[n][oc][i / output_W][i % output_W] += 
                        filter_col[oc][j] * im_col[n][j][i];
                }
            }
        }
    }
}

// Naive 2D convolution implementation
void naive_conv2d() {
    // Initialize output_naive to 0
    memset(output_naive, 0, sizeof(output_naive));

    // Perform naive convolution
    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < output_channels; ++oc) {
            for (int h_out = 0; h_out < output_H; ++h_out) {
                for (int w_out = 0; w_out < output_W; ++w_out) {
                    for (int ic = 0; ic < input_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int h_in = h_out * stride + kh - padding;
                                int w_in = w_out * stride + kw - padding;
                                if (h_in >= 0 && h_in < feature_H && w_in >= 0 && w_in < feature_W) {
                                    output_feature_map[n][oc][h_out][w_out] +=
                                        input_feature_map[n][ic][h_in][w_in] * filter[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void initialize() {
    // Initialize input feature map
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < input_channels; ++c) {
            for (int h = 0; h < feature_H; ++h) {
                for (int w = 0; w < feature_W; ++w) {
                    input_feature_map[n][c][h][w] = rand() % 10; // Random values for input feature map
                }
            }
        }
    }

    // Initialize filters
    for (int oc = 0; oc < output_channels; ++oc) {
        for (int ic = 0; ic < input_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    filter[oc][ic][kh][kw] = rand() % 10; // Random values for filter
                }
            }
        }
    }
    //naive_conv2d(); // ground truth
}

// Test function to compare im2col result and naive conv result
void test() {
    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < output_channels; ++oc) {
            for (int h = 0; h < output_H; ++h) {
                for (int w = 0; w < output_W; ++w) {
                    assert(output_feature_map[n][oc][h][w] == output_naive[n][oc][h][w]);
                }
            }
        }
    }
}

void im2col(){
    input2col();
    filter2col();
    matmul();
}

int main() {
    // Initialize input feature map and filters with some values for testing
    initialize();
    
    float avg_time = 0.0f;
    for (int k = 0; k < 32; ++k) {
        auto start_time = get_time();
        im2col();
        //naive_conv2d();
        //test();
        printf("%f\n", get_time() - start_time);
        avg_time += get_time() - start_time;
    }
    std::cout << "Average Time is: " << (avg_time / 32) << " seconds" << std::endl;
    return 0;
}
