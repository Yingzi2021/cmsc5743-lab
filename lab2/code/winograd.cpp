#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

const int m = 2; // Output tile size
const int r = 3; // Kernel size
const int alpha = m + r - 1; // Transformed matrix size
const int H = 56; // Input height
const int W = 56; // Input width
const int K = 64; // Number of output channels
const int C = 3; // Number of input channels
const int N = 1; // Batch size
const int tile_h = static_cast<int>(ceil(static_cast<float>(H) / m));
const int tile_w = static_cast<int>(ceil(static_cast<float>(W) / m));
const int P = N * tile_h * tile_w; // Number of tiles

// 矩阵乘法 (a * b) 结果保存在 c 中
void matrix_multiply(const vector<vector<float>> &a, const vector<vector<float>> &b, vector<vector<float>> &c) {
    int rows = a.size();
    int cols = b[0].size();
    int inner = a[0].size();
    c.assign(rows, vector<float>(cols, 0.0f));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < inner; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// 矩阵转置
void matrix_transpose(const vector<vector<float>> &a, vector<vector<float>> &result) {
    int rows = a.size();
    int cols = a[0].size();
    result.assign(cols, vector<float>(rows));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = a[i][j];
        }
    }
}

// 生成随机输入和卷积核
void generate_random_input_and_kernel(vector<vector<vector<vector<int>>>> &D, vector<vector<vector<vector<int>>>> &g) {
    D.resize(N, vector<vector<vector<int>>>(C, vector<vector<int>>(H, vector<int>(W))));
    g.resize(K, vector<vector<vector<int>>>(C, vector<vector<int>>(r, vector<int>(r))));

    srand(static_cast<unsigned>(time(0)));
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    D[n][c][i][j] = rand() % 10;
                }
            }
        }
    }
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < r; j++) {
                    g[k][c][i][j] = (rand() % 3) - 1; // [-1, 1]
                }
            }
        }
    }
}

// 朴素卷积实现
void naive_convolution(const vector<vector<vector<vector<int>>>> &D, const vector<vector<vector<vector<int>>>> &g, vector<vector<vector<vector<int>>>> &Y_naive) {
    int out_height = H - r + 1;
    int out_width = W - r + 1;
    Y_naive.resize(N, vector<vector<vector<int>>>(K, vector<vector<int>>(out_height, vector<int>(out_width, 0))));

    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            for (int c = 0; c < C; c++) {
                for (int i = 0; i < out_height; i++) {
                    for (int j = 0; j < out_width; j++) {
                        int sum = 0;
                        for (int p = 0; p < r; p++) {
                            for (int q = 0; q < r; q++) {
                                sum += D[n][c][i + p][j + q] * g[k][c][p][q];
                            }
                        }
                        Y_naive[n][k][i][j] += sum;
                    }
                }
            }
        }
    }
}

// Winograd卷积实现
void winograd_convolution(const vector<vector<vector<vector<int>>>> &D, const vector<vector<vector<vector<int>>>> &g, vector<vector<vector<vector<int>>>> &Y_winograd) {
    // 初始化Winograd变换矩阵
    vector<vector<float>> G = {
        {1, 0, 0},
        {0.5f, 0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f},
        {0, 0, 1}
    };
    vector<vector<float>> B = {
        {1, 0, 0, 0},
        {0, 1, -1, 1},
        {-1, 1, 1, 0},
        {0, 0, 0, -1}
    };
    vector<vector<float>> A = {
        {1, 0},
        {1, 1},
        {1, -1},
        {0, -1}
    };

    // 计算变换后的卷积核U
    vector<vector<vector<vector<float>>>> U(alpha, vector<vector<vector<float>>>(alpha, vector<vector<float>>(K, vector<float>(C, 0.0f))));
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            vector<vector<float>> g_kc(r, vector<float>(r));
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < r; j++) {
                    g_kc[i][j] = static_cast<float>(g[k][c][i][j]);
                }
            }
            vector<vector<float>> temp(alpha, vector<float>(r));
            vector<vector<float>> u(alpha, vector<float>(alpha));

            // G * g_kc
            matrix_multiply(G, g_kc, temp);
            // temp * G^T
            vector<vector<float>> G_T;
            matrix_transpose(G, G_T);
            matrix_multiply(temp, G_T, u);

            for (int xi = 0; xi < alpha; xi++) {
                for (int nu = 0; nu < alpha; nu++) {
                    U[xi][nu][k][c] = u[xi][nu];
                }
            }
        }
    }

    // 计算变换后的输入V
    vector<vector<vector<vector<float>>>> V(alpha, vector<vector<vector<float>>>(alpha, vector<vector<float>>(C, vector<float>(P, 0.0f))));
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int y = 0; y < tile_h; y++) {
                for (int x = 0; x < tile_w; x++) {
                    vector<vector<float>> d(alpha, vector<float>(alpha, 0.0f));
                    for (int i = 0; i < alpha; i++) {
                        for (int j = 0; j < alpha; j++) {
                            int src_row = y * m + i;
                            int src_col = x * m + j;
                            if (src_row < H && src_col < W) {
                                d[i][j] = static_cast<float>(D[n][c][src_row][src_col]);
                            } else {
                                d[i][j] = 0.0f;
                            }
                        }
                    }

                    // B^T * d * B
                    vector<vector<float>> temp(alpha, vector<float>(alpha));
                    vector<vector<float>> B_T;
                    matrix_transpose(B, B_T);
                    vector<vector<float>> temp2(alpha, vector<float>(alpha));
                    matrix_multiply(B_T, d, temp);
                    matrix_multiply(temp, B, temp2);

                    int b = n * (tile_h * tile_w) + y * tile_w + x;
                    for (int xi = 0; xi < alpha; xi++) {
                        for (int nu = 0; nu < alpha; nu++) {
                            V[xi][nu][c][b] = temp2[xi][nu];
                        }
                    }
                }
            }
        }
    }

    // 元素级乘法得到M
    vector<vector<vector<vector<float>>>> M(alpha, vector<vector<vector<float>>>(alpha, vector<vector<float>>(K, vector<float>(P, 0.0f))));
    for (int xi = 0; xi < alpha; xi++) {
        for (int nu = 0; nu < alpha; nu++) {
            for (int k = 0; k < K; k++) {
                for (int p = 0; p < P; p++) {
                    float sum = 0.0f;
                    for (int c = 0; c < C; c++) {
                        sum += U[xi][nu][k][c] * V[xi][nu][c][p];
                    }
                    M[xi][nu][k][p] = sum;
                }
            }
        }
    }

    // 反变换得到输出Y_winograd
    int out_height = H - r + 1;
    int out_width = W - r + 1;
    Y_winograd.resize(N, vector<vector<vector<int>>>(K, vector<vector<int>>(out_height, vector<int>(out_width, 0))));

    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            for (int y = 0; y < tile_h; y++) {
                for (int x = 0; x < tile_w; x++) {
                    int b = n * (tile_h * tile_w) + y * tile_w + x;
                    vector<vector<float>> temp_m(alpha, vector<float>(alpha));
                    for (int xi = 0; xi < alpha; xi++) {
                        for (int nu = 0; nu < alpha; nu++) {
                            temp_m[xi][nu] = M[xi][nu][k][b];
                        }
                    }

                    // A^T * temp_m * A
                    vector<vector<float>> temp1(m, vector<float>(alpha));
                    vector<vector<float>> Y_tile(m, vector<float>(m));
                    vector<vector<float>> A_T;
                    matrix_transpose(A, A_T);
                    matrix_multiply(A_T, temp_m, temp1);
                    matrix_multiply(temp1, A, Y_tile);

                    // 将结果写回Y_winograd
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < m; j++) {
                            int dst_row = y * m + i;
                            int dst_col = x * m + j;
                            if (dst_row < out_height && dst_col < out_width) {
                                Y_winograd[n][k][dst_row][dst_col] = static_cast<int>(round(Y_tile[i][j]));
                            }
                        }
                    }
                }
            }
        }
    }
}

// 比较结果是否一致
bool compare_results(const vector<vector<vector<vector<int>>>> &Y_naive, const vector<vector<vector<vector<int>>>> &Y_winograd) {
    int N = Y_naive.size();
    int K = Y_naive[0].size();
    int H = Y_naive[0][0].size();
    int W = Y_naive[0][0][0].size();

    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    if (abs(Y_naive[n][k][i][j] - Y_winograd[n][k][i][j]) > 1e-5) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

int main() {
    // 生成随机输入和卷积核
    vector<vector<vector<vector<int>>>> D;
    vector<vector<vector<vector<int>>>> g;
    generate_random_input_and_kernel(D, g);

    // 进行朴素卷积
    vector<vector<vector<vector<int>>>> Y_naive;
    naive_convolution(D, g, Y_naive);

    // 进行Winograd卷积
    vector<vector<vector<vector<int>>>> Y_winograd;
    winograd_convolution(D, g, Y_winograd);

    // 比较结果
    if (compare_results(Y_naive, Y_winograd)) {
        cout << "朴素卷积和Winograd卷积的结果一致！" << endl;
    } else {
        cout << "朴素卷积和Winograd卷积的结果不一致！" << endl;
    }

    return 0;
}
