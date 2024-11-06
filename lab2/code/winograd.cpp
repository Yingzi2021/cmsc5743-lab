#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Configuration constants for convolution parameters
constexpr int batch = 1;              // Number of batches
constexpr int feature_H = 56;         // Height of the input feature map
constexpr int feature_W = 56;         // Width of the input feature map
constexpr int input_channels = 3;     // Number of input channels
constexpr int output_channels = 64;   // Number of output channels
constexpr int kernel_size = 3;        // Size of the convolution kernel
constexpr int stride = 1;             // Stride of the convolution
constexpr int padding = 0;            // Padding size

// Calculated dimensions for output feature map
constexpr int output_H = feature_H - kernel_size + 1;  // Output height
constexpr int output_W = feature_W - kernel_size + 1;  // Output width

// Winograd parameters
constexpr int m = 2;                  // Output tile size for Winograd
constexpr int r = kernel_size;        // Kernel size
constexpr int alpha = m + r - 1;      // Size of transformed matrix
constexpr int tile_h = static_cast<int>(ceil(static_cast<float>(feature_H) / m));  // Tile height
constexpr int tile_w = static_cast<int>(ceil(static_cast<float>(feature_W) / m));  // Tile width
constexpr int P = batch * tile_h * tile_w;  // Total number of tiles

// Input feature maps, filters, and output maps for various methods
int input_feature_map[batch][input_channels][feature_H][feature_W];
int filter[output_channels][input_channels][kernel_size][kernel_size];
int output_im2col[batch][output_channels][output_H][output_W];
int output_winograd[batch][output_channels][output_H][output_W];
int output_naive[batch][output_channels][output_H][output_W];

// Temporary arrays for im2col operation
int im_col[batch][input_channels * kernel_size * kernel_size][output_H * output_W];
int filter_col[output_channels][input_channels * kernel_size * kernel_size];

// Get current time in seconds (for benchmarking)
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// Generate random values for input feature maps and filters
void generate_random_input_and_kernel() {
    srand(static_cast<unsigned>(time(0)));
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < input_channels; c++) {
            for (int i = 0; i < feature_H; i++) {
                for (int j = 0; j < feature_W; j++) {
                    input_feature_map[n][c][i][j] = rand() % 10;
                }
            }
        }
    }
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

// Convert input feature map to column format for im2col convolution
void input2col() {
    for (int n = 0; n < batch; ++n) {
        for (int h_out = 0; h_out < output_H; ++h_out) {
            for (int w_out = 0; w_out < output_W; ++w_out) {
                int col_index = h_out * output_W + w_out;
                for (int c = 0; c < input_channels; ++c) {
                    int channel_offset = c * kernel_size * kernel_size;
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int h_in = h_out * stride + kh - padding;
                            int w_in = w_out * stride + kw - padding;
                            if (h_in >= 0 && h_in < feature_H && w_in >= 0 && w_in < feature_W) {
                                im_col[n][channel_offset + kh * kernel_size + kw][col_index] =
                                    input_feature_map[n][c][h_in][w_in];
                            } else {
                                im_col[n][channel_offset + kh * kernel_size + kw][col_index] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Convert filters to column format for im2col convolution
void filter2col() {
    for (int oc = 0; oc < output_channels; ++oc) {
        for (int ic = 0; ic < input_channels; ++ic) {
            int col_index = ic * kernel_size * kernel_size;
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    filter_col[oc][col_index + kh * kernel_size + kw] = filter[oc][ic][kh][kw];
                }
            }
        }
    }
}

// Matrix multiplication to compute output feature map from im2col arrays
void matmul() {
    memset(output_im2col, 0, sizeof(output_im2col));
    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < output_channels; ++oc) {
            for (int i = 0; i < output_H * output_W; ++i) {
                for (int j = 0; j < input_channels * kernel_size * kernel_size; ++j) {
                    output_im2col[n][oc][i / output_W][i % output_W] +=
                        filter_col[oc][j] * im_col[n][j][i];
                }
            }
        }
    }
}

// Perform im2col-based convolution
void im2col_convolution() {
    input2col();
    filter2col();
    matmul();
}

// Implementation of Winograd convolution algorithm
void winograd_convolution() {
    // Initialize transformation matrices for Winograd convolution
    // Matrix G transforms the filter
    float G[alpha][r] = {
        {1, 0, 0},
        {0.5f, 0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f},
        {0, 0, 1}
    };

    // Matrix B transforms the input feature map
    float B[alpha][alpha] = {
        {1, 0, 0, 0},
        {0, 1, -1, 1},
        {-1, 1, 1, 0},
        {0, 0, 0, -1}
    };

    // Matrix A is used for the inverse transformation to obtain the final output
    float A[alpha][m] = {
        {1, 0},
        {1, 1},
        {1, -1},
        {0, -1}
    };

    // Pre-compute the transformed filter matrix U
    float U[alpha][alpha][output_channels][input_channels];
    for (int k = 0; k < output_channels; k++) {
        for (int c = 0; c < input_channels; c++) {
            // Copy the filter values for the current output and input channel
            float g_kc[r][r];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < r; j++)
                    g_kc[i][j] = static_cast<float>(filter[k][c][i][j]);

            // Compute G * g_kc * G^T for the transformed filter
            float temp[alpha][r];
            for (int i = 0; i < alpha; i++) {
                for (int j = 0; j < r; j++) {
                    temp[i][j] = 0.0f;
                    for (int k1 = 0; k1 < r; k1++)
                        temp[i][j] += G[i][k1] * g_kc[k1][j];
                }
            }
            float u[alpha][alpha];
            for (int i = 0; i < alpha; i++) {
                for (int j = 0; j < alpha; j++) {
                    u[i][j] = 0.0f;
                    for (int k1 = 0; k1 < r; k1++)
                        u[i][j] += temp[i][k1] * G[k1][j];
                }
            }

            // Store the transformed filter matrix U
            for (int xi = 0; xi < alpha; xi++)
                for (int nu = 0; nu < alpha; nu++)
                    U[xi][nu][k][c] = u[xi][nu];
        }
    }

    // Compute transformed input V
    float V[alpha][alpha][input_channels][P];
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < input_channels; c++) {
            for (int y = 0; y < tile_h; y++) {
                for (int x = 0; x < tile_w; x++) {
                    float d[alpha][alpha] = {0};  // Input tile
                    for (int i = 0; i < alpha; i++) {
                        for (int j = 0; j < alpha; j++) {
                            int src_row = y * m + i;
                            int src_col = x * m + j;
                            d[i][j] = (src_row < feature_H && src_col < feature_W)
                                      ? static_cast<float>(input_feature_map[n][c][src_row][src_col])
                                      : 0.0f;
                        }
                    }

                    // Apply B^T * d * B
                    float temp[alpha][alpha];
                    for (int i = 0; i < alpha; i++)
                        for (int j = 0; j < alpha; j++) {
                            temp[i][j] = 0.0f;
                            for (int k1 = 0; k1 < alpha; k1++)
                                temp[i][j] += B[i][k1] * d[k1][j];
                        }

                    float v[alpha][alpha];
                    for (int i = 0; i < alpha; i++)
                        for (int j = 0; j < alpha; j++) {
                            v[i][j] = 0.0f;
                            for (int k1 = 0; k1 < alpha; k1++)
                                v[i][j] += temp[i][k1] * B[k1][j];
                        }

                    int b = n * (tile_h * tile_w) + y * tile_w + x;
                    for (int xi = 0; xi < alpha; xi++)
                        for (int nu = 0; nu < alpha; nu++)
                            V[xi][nu][c][b] = v[xi][nu];
                }
            }
        }
    }

    // Element-wise multiplication of U and V to get M
    float M[alpha][alpha][output_channels][P];
    for (int xi = 0; xi < alpha; xi++) {
        for (int nu = 0; nu < alpha; nu++) {
            for (int k = 0; k < output_channels; k++) {
                for (int p = 0; p < P; p++) {
                    float sum = 0.0f;
                    for (int c = 0; c < input_channels; c++) {
                        sum += U[xi][nu][k][c] * V[xi][nu][c][p];
                    }
                    M[xi][nu][k][p] = sum;
                }
            }
        }
    }

    // Inverse transform of M to get the final output
    memset(output_winograd, 0, sizeof(output_winograd));
    for (int n = 0; n < batch; n++) {
        for (int k = 0; k < output_channels; k++) {
            for (int y = 0; y < tile_h; y++) {
                for (int x = 0; x < tile_w; x++) {
                    int b = n * (tile_h * tile_w) + y * tile_w + x;

                    // A^T * M * A to get output tile Y
                    float temp_m[alpha][alpha];
                    for (int xi = 0; xi < alpha; xi++)
                        for (int nu = 0; nu < alpha; nu++)
                            temp_m[xi][nu] = M[xi][nu][k][b];

                    float temp1[m][alpha];
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < alpha; j++) {
                            temp1[i][j] = 0.0f;
                            for (int k1 = 0; k1 < alpha; k1++)
                                temp1[i][j] += A[i][k1] * temp_m[k1][j];
                        }

                    float Y_tile[m][m];
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < m; j++) {
                            Y_tile[i][j] = 0.0f;
                            for (int k1 = 0; k1 < alpha; k1++)
                                Y_tile[i][j] += temp1[i][k1] * A[k1][j];
                        }

                    // Copy Y_tile back to the output feature map
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < m; j++) {
                            int dst_row = y * m + i;
                            int dst_col = x * m + j;
                            if (dst_row < output_H && dst_col < output_W) {
                                output_winograd[n][k][dst_row][dst_col] = static_cast<int>(round(Y_tile[i][j]));
                            }
                        }
                    }
                }
            }
        }
    }
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

int main() {
    generate_random_input_and_kernel();  // Generate input data
    naive_convolution();  // Compute naive convolution

    // Measure time taken by im2col convolution method
    float avg_time_im2col = 0.0f;
    for (int k = 0; k < 32; ++k) {
        auto start_time = get_time();
        im2col_convolution();
        test(output_naive, output_im2col);  // Verify result matches naive convolution
        avg_time_im2col += get_time() - start_time;
    }
    std::cout << "Average Time for im2col: " << (avg_time_im2col / 32) << " seconds" << std::endl;

    // Measure time taken by Winograd convolution method
    float avg_time_winograd = 0.0f;
    for (int k = 0; k < 32; ++k) {
        auto start_time = get_time();
        winograd_convolution();
        test(output_naive, output_winograd);  // Verify result matches naive convolution
        avg_time_winograd += get_time() - start_time;
    }
    std::cout << "Average Time for Winograd: " << (avg_time_winograd / 32) << " seconds" << std::endl;

    return 0;
}

