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
const int P = N * static_cast<int>(ceil(static_cast<float>(H) / m)) * static_cast<int>(ceil(static_cast<float>(W) / m)); // Number of tiles

// Helper function to initialize matrices
void initialize_matrix(vector<vector<float>> &matrix, const vector<vector<float>> &values) {
    for (int i = 0; i < values.size(); i++) {
        for (int j = 0; j < values[i].size(); j++) {
            matrix[i][j] = values[i][j];
        }
    }
}

// Matrix multiplication (a * b) result in c
void matrix_multiply(const vector<vector<float>> &a, const vector<vector<float>> &b, vector<vector<float>> &c) {
    int rows = a.size();
    int cols = b[0].size();
    int inner = a[0].size();
    c.resize(rows, vector<float>(cols, 0.0));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < inner; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// Matrix transpose
void matrix_transpose(const vector<vector<float>> &a, vector<vector<float>> &result) {
    int rows = a.size();
    int cols = a[0].size();
    result.resize(cols, vector<float>(rows));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = a[i][j];
        }
    }
}

// Padding to ensure 4x4 dimensions
void pad_to_4x4(vector<vector<int>> &matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    if (rows < alpha || cols < alpha) {
        matrix.resize(alpha, vector<int>(alpha, 0));
        for (int i = rows; i < alpha; i++) {
            for (int j = cols; j < alpha; j++) {
                matrix[i][j] = 0;
            }
        }
    }
}

// Function to generate random input and kernel
void generate_random_input_and_kernel(vector<vector<vector<int>>> &D, vector<vector<vector<vector<int>>>> &g) {
    D.resize(N, vector<vector<int>>(C, vector<int>(H, vector<int>(W, 0))));
    g.resize(K, vector<vector<vector<int>>>(C, vector<vector<int>>(r, vector<int>(r, 0))));

    srand(static_cast<unsigned>(time(0)));
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < H + 2; i++) {
                for (int j = 0; j < W + 2; j++) {
                    D[n][c][i][j] = rand() % 10;
                }
            }
        }
    }
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < r; j++) {
                    g[k][c][i][j] = (rand() % 3) - 1;
                }
            }
        }
    }
}

// Naive convolution implementation
void naive_convolution(const vector<vector<vector<int>>> &D, const vector<vector<vector<vector<int>>>> &g, vector<vector<vector<int>>> &Y_naive) {
    int out_height = H - r + 1;
    int out_width = W - r + 1;
    Y_naive.resize(N, vector<vector<int>>(K, vector<int>(out_height, vector<int>(out_width, 0))));

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

// Winograd convolution implementation
void winograd_convolution(const vector<vector<vector<int>>> &D, const vector<vector<vector<vector<int>>>> &g, vector<vector<vector<int>>> &Y_winograd) {
    // Initialize Winograd transform matrices
    vector<vector<float>> G = {{1, 0, 0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0, 0, 1}};
    vector<vector<float>> B = {{1, 0, 0, 0}, {0, 1, -1, 1}, {-1, 1, 1, 0}, {0, 0, 0, -1}};
    vector<vector<float>> A = {{1, 0}, {1, 1}, {1, -1}, {0, -1}};

    // Transformed kernel U
    vector<vector<vector<vector<float>>>> U(K, vector<vector<vector<float>>>(C, vector<vector<float>>(alpha, vector<float>(alpha, 0))));
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            vector<vector<float>> u(alpha, vector<float>(alpha, 0));
            matrix_multiply(G, g[k][c], u);
            matrix_multiply(u, G, U[k][c]);
        }
    }

    // Transformed input V
    vector<vector<vector<vector<float>>>> V(alpha, vector<vector<vector<float>>>(alpha, vector<vector<float>>(C, vector<float>(P, 0))));
    for (int i = 0; i < N; i++) {
        for (int c = 0; c < C; c++) {
            for (int y = 0; y < ceil(H / m); y++) {
                for (int x = 0; x < ceil(W / m); x++) {
                    vector<vector<int>> d(alpha, vector<int>(alpha, 0));
                    for (int row = 0; row < alpha; row++) {
                        for (int col = 0; col < alpha; col++) {
                            int src_row = y * m + row;
                            int src_col = x * m + col;
                            if (src_row < H && src_col < W) {
                                d[row][col] = D[i][c][src_row][src_col];
                            }
                        }
                    }
                    pad_to_4x4(d); // Ensure d is 4x4 if on the boundary
                    vector<vector<float>> v(alpha, vector<float>(alpha, 0));
                    matrix_multiply(B, d, v);
                    matrix_multiply(v, B, V[c][y * (W / m) + x]);
                }
            }
        }
    }

    // Element-wise multiplication of U and V to get M
    vector<vector<vector<vector<float>>>> M(alpha, vector<vector<vector<float>>>(alpha, vector<vector<float>>(K, vector<float>(P, 0))));
    for (int xi = 0; xi < alpha; xi++) {
        for (int nu = 0; nu < alpha; nu++) {
            for (int k = 0; k < K; k++) {
                for (int p = 0; p < P; p++) {
                    M[xi][nu][k][p] = U[k][xi][nu] * V[xi][nu][k][p];
                }
            }
        }
    }

    // Transform M back using A to get output Y_winograd
Y_winograd.resize(N, vector<vector<int>>(K, vector<int>(H, vector<int>(W, 0))));
for (int i = 0; i < N; i++) {
    for (int k = 0; k < K; k++) {
        for (int y = 0; y < ceil(H / m); y++) {
            for (int x = 0; x < ceil(W / m); x++) {
                int tile_idx = y * (W / m) + x;
                vector<vector<float>> temp_m(alpha, vector<float>(alpha, 0));

                // Extract the corresponding M tile
                for (int xi = 0; xi < alpha; xi++) {
                    for (int nu = 0; nu < alpha; nu++) {
                        temp_m[xi][nu] = M[xi][nu][k][tile_idx];
                    }
                }

                // Perform A.T * temp_m * A transformation
                vector<vector<float>> Y_tile(m, vector<float>(m, 0));
                vector<vector<float>> temp_a(alpha, vector<float>(m, 0));
                matrix_multiply(temp_m, A, temp_a);
                matrix_multiply(A, temp_a, Y_tile);

                // Write the transformed tile back to Y_winograd at the correct position
                for (int row = 0; row < m; row++) {
                    for (int col = 0; col < m; col++) {
                        int dst_row = y * m + row;
                        int dst_col = x * m + col;
                        if (dst_row < H && dst_col < W) {
                            Y_winograd[i][k][dst_row][dst_col] = static_cast<int>(Y_tile[row][col]);
                        }
                    }
                }
            }
        }
    }
    }
}

bool compare_results(const vector<vector<vector<int>>> &Y_naive, const vector<vector<vector<int>>> &Y_winograd) {
    for (int i = 0; i < Y_naive.size(); i++) {
        for (int j = 0; j < Y_naive[0].size(); j++) {
            for (int x = 0; x < Y_naive[0][0].size(); x++) {
                for (int y = 0; y < Y_naive[0][0][0].size(); y++) {
                    if (abs(Y_naive[i][j][x][y] - Y_winograd[i][j][x][y]) > 1e-5) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

int main() {
    // Initialize Winograd transform matrices
    vector<vector<float>> G = {{1, 0, 0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0, 0, 1}};
    vector<vector<float>> B = {{1, 0, 0, 0}, {0, 1, -1, 1}, {-1, 1, 1, 0}, {0, 0, 0, -1}};
    vector<vector<float>> A = {{1, 0}, {1, 1}, {1, -1}, {0, -1}};

    // Random input and kernel generation
    vector<vector<vector<int>>> D;
    vector<vector<vector<vector<int>>>> g;
    generate_random_input_and_kernel(D, g);

    // Perform naive convolution
    vector<vector<vector<int>>> Y_naive;
    naive_convolution(D, g, Y_naive);

    // Perform Winograd convolution
    vector<vector<vector<int>>> Y_winograd;
    winograd_convolution(D, g, Y_winograd);

    // Compare results
    if (compare_results(Y_naive, Y_winograd)) {
        cout << "The results from Naive and Winograd convolutions are approximately equal!" << endl;
    } else {
        cout << "The results from Naive and Winograd convolutions differ." << endl;
    }

    return 0;
}
    