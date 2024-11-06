#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>

constexpr int batch = 1;
constexpr int feature_H = 56;
constexpr int feature_W = 56;
constexpr int input_channels = 3;
constexpr int output_channels = 64;
constexpr int kernel_size = 3;
constexpr int stride = 1;
constexpr int padding = 0;

constexpr int output_H = feature_H - kernel_size + 1;
constexpr int output_W = feature_W - kernel_size + 1;

// Winograd 参数
constexpr int m = 2;  // Output tile size
constexpr int r = kernel_size;  // Kernel size
constexpr int alpha = m + r - 1;  // Transformed matrix size
constexpr int tile_h = static_cast<int>(ceil(static_cast<float>(feature_H) / m));
constexpr int tile_w = static_cast<int>(ceil(static_cast<float>(feature_W) / m));
constexpr int P = batch * tile_h * tile_w;  // Number of tiles

int input_feature_map[batch][input_channels][feature_H][feature_W];
int filter[output_channels][input_channels][kernel_size][kernel_size];
int output_im2col[batch][output_channels][output_H][output_W];
int output_winograd[batch][output_channels][output_H][output_W];
int output_naive[batch][output_channels][output_H][output_W];

int im_col[batch][input_channels * kernel_size * kernel_size][output_H * output_W];  // Convert input feature map to cols
int filter_col[output_channels][input_channels * kernel_size * kernel_size];  // Convert filter to cols

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// 生成随机输入和卷积核
void generate_random_input_and_kernel() {
    srand(static_cast<unsigned>(time(0)));
    // 初始化输入特征图
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < input_channels; c++) {
            for (int i = 0; i < feature_H; i++) {
                for (int j = 0; j < feature_W; j++) {
                    input_feature_map[n][c][i][j] = rand() % 10;
                }
            }
        }
    }
    // 初始化卷积核
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

// 朴素卷积实现
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

// im2col 方法实现卷积
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
    memset(output_im2col, 0, sizeof(output_im2col));

    // Perform matrix multiplication
    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < output_channels; ++oc) {
            for (int i = 0; i < output_H * output_W; ++i) {  // Iterate over output feature map positions
                for (int j = 0; j < input_channels * kernel_size * kernel_size; ++j) {  // Iterate over filter_col and im_col
                    output_im2col[n][oc][i / output_W][i % output_W] +=
                        filter_col[oc][j] * im_col[n][j][i];  // perform output column to matrix by using index `i / output_W` and `i % output_W`
                }
            }
        }
    }
}

void im2col_convolution() {
    input2col();
    filter2col();
    matmul();
}

// Winograd 卷积实现
void winograd_convolution() {
    // 初始化Winograd变换矩阵
    float G[alpha][r] = {
        {1, 0, 0},
        {0.5f, 0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f},
        {0, 0, 1}
    };
    float GT[r][alpha];
    for (int i = 0; i < alpha; i++)
        for (int j = 0; j < r; j++)
            GT[j][i] = G[i][j];

    float B[alpha][alpha] = {
        {1, 0, 0, 0},
        {0, 1, -1, 1},
        {-1, 1, 1, 0},
        {0, 0, 0, -1}
    };
    float BT[alpha][alpha];
    for (int i = 0; i < alpha; i++)
        for (int j = 0; j < alpha; j++)
            BT[j][i] = B[i][j];

    float A[alpha][m] = {
        {1, 0},
        {1, 1},
        {1, -1},
        {0, -1}
    };
    float AT[m][alpha];
    for (int i = 0; i < alpha; i++)
        for (int j = 0; j < m; j++)
            AT[j][i] = A[i][j];

    // 计算变换后的卷积核U
    float U[alpha][alpha][output_channels][input_channels];
    for (int k = 0; k < output_channels; k++) {
        for (int c = 0; c < input_channels; c++) {
            float g_kc[r][r];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < r; j++)
                    g_kc[i][j] = static_cast<float>(filter[k][c][i][j]);

            float temp[alpha][r];
            for (int i = 0; i < alpha; i++)
                for (int j = 0; j < r; j++) {
                    temp[i][j] = 0.0f;
                    for (int k1 = 0; k1 < r; k1++)
                        temp[i][j] += G[i][k1] * g_kc[k1][j];
                }

            float u[alpha][alpha];
            for (int i = 0; i < alpha; i++)
                for (int j = 0; j < alpha; j++) {
                    u[i][j] = 0.0f;
                    for (int k1 = 0; k1 < r; k1++)
                        u[i][j] += temp[i][k1] * GT[k1][j];
                }

            for (int xi = 0; xi < alpha; xi++)
                for (int nu = 0; nu < alpha; nu++)
                    U[xi][nu][k][c] = u[xi][nu];
        }
    }

    // 计算变换后的输入V
    float V[alpha][alpha][input_channels][P];
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < input_channels; c++) {
            for (int y = 0; y < tile_h; y++) {
                for (int x = 0; x < tile_w; x++) {
                    float d[alpha][alpha] = {0};
                    for (int i = 0; i < alpha; i++) {
                        for (int j = 0; j < alpha; j++) {
                            int src_row = y * m + i;
                            int src_col = x * m + j;
                            if (src_row < feature_H && src_col < feature_W) {
                                d[i][j] = static_cast<float>(input_feature_map[n][c][src_row][src_col]);
                            } else {
                                d[i][j] = 0.0f;
                            }
                        }
                    }

                    // B^T * d * B
                    float temp[alpha][alpha];
                    for (int i = 0; i < alpha; i++)
                        for (int j = 0; j < alpha; j++) {
                            temp[i][j] = 0.0f;
                            for (int k1 = 0; k1 < alpha; k1++)
                                temp[i][j] += BT[i][k1] * d[k1][j];
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

    // 元素级乘法得到M
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

    // 反变换得到输出
    memset(output_winograd, 0, sizeof(output_winograd));
    for (int n = 0; n < batch; n++) {
        for (int k = 0; k < output_channels; k++) {
            for (int y = 0; y < tile_h; y++) {
                for (int x = 0; x < tile_w; x++) {
                    int b = n * (tile_h * tile_w) + y * tile_w + x;
                    float temp_m[alpha][alpha];
                    for (int xi = 0; xi < alpha; xi++)
                        for (int nu = 0; nu < alpha; nu++)
                            temp_m[xi][nu] = M[xi][nu][k][b];

                    // A^T * temp_m * A
                    float temp1[m][alpha];
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < alpha; j++) {
                            temp1[i][j] = 0.0f;
                            for (int k1 = 0; k1 < alpha; k1++)
                                temp1[i][j] += AT[i][k1] * temp_m[k1][j];
                        }

                    float Y_tile[m][m];
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < m; j++) {
                            Y_tile[i][j] = 0.0f;
                            for (int k1 = 0; k1 < alpha; k1++)
                                Y_tile[i][j] += temp1[i][k1] * A[k1][j];
                        }

                    // 将结果写回output_winograd
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

// 比较结果是否一致
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
    generate_random_input_and_kernel();
    naive_convolution();

    float avg_time_im2col = 0.0f;
    for (int k = 0; k < 32; ++k) {
        auto start_time = get_time();
        im2col_convolution();
        test(output_naive, output_im2col);
        printf("%f\n", get_time() - start_time);
        avg_time_im2col += get_time() - start_time;
    }
    std::cout << "Average Time for im2col: " << (avg_time_im2col / 32) << " seconds" << std::endl;

    float avg_time_winograd = 0.0f;
    for (int k = 0; k < 32; ++k) {
        auto start_time = get_time();
        winograd_convolution();
        test(output_naive, output_winograd);
        printf("%f\n", get_time() - start_time);
        avg_time_winograd += get_time() - start_time;
    }
    std::cout << "Average Time for Winograd: " << (avg_time_winograd / 32) << " seconds" << std::endl;

    return 0;
}
