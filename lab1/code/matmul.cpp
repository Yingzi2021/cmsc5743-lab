// g++ matmul.cpp -o matmul -std=c++17 -O3 -Wall && ./matmul

#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>
#include <immintrin.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

constexpr int n = 1024;
int A[n][n]; // Matrix A
int B[n][n]; // Matrix B
int BT[n][n]; // transpose
int AT[n][n];
int C[n][n]; // Matrix C (result of A * B)
int C_groundtruth[n][n]; // correct result of 

void init() {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = rand(); 
      B[i][j] = rand(); 
    } 
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C_groundtruth[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void test() {// test if the result is right
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      assert(C[i][j] == C_groundtruth[i][j]);
    }
  }
}

void matmul() { // naive
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}

void matmul_ikj() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) {
      for (int j = 0; j < n; j++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}

void matmul_AT() {
  memset(C, 0, sizeof(C));
  // do transpose
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      AT[i][j] = A[j][i];
    }
  }
  //do calculation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += AT[k][i] * B[k][j]; 
      }   
    }
  }
}

void matmul_BT() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      BT[i][j] = B[j][i];
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * BT[j][k];    
      }   
    }
  }
}

// matrix multiplication with loop unrolling
void matmul_unrolled() {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            int sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
            for (int k = 0; k < n; k += 8) {
                sum0 += A[i][k] * B[k][j];
                sum1 += A[i][k + 1] * B[k + 1][j];
                sum2 += A[i][k + 2] * B[k + 2][j];
                sum3 += A[i][k + 3] * B[k + 3][j];
                sum4 += A[i][k + 4] * B[k + 4][j];
                sum5 += A[i][k + 5] * B[k + 5][j];
                sum6 += A[i][k + 6] * B[k + 6][j];
                sum7 += A[i][k + 7] * B[k + 7][j];
            }
            C[i][j] = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
        }
    }
}

void matmul_simd() {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Create an accumulator register for C[i][j]
            __m256i c_vec = _mm256_setzero_si256();  // Initialize c_vec to 0            
            for (int k = 0; k < n; k += 8) {
                // Load A[i][k:k+7] and B[k:k+7][j] into SIMD registers
                __m256i a_vec = _mm256_loadu_si256((__m256i*)&A[i][k]);  // Load A[i][k:k+7]
                // Load the column vector of matrix B
                __m256i b_vec = _mm256_set_epi32(
                    B[k+7][j], B[k+6][j], B[k+5][j], B[k+4][j], 
                    B[k+3][j], B[k+2][j], B[k+1][j], B[k][j]);
                // Perform vectorized multiplication and accumulate into c_vec
                __m256i mul_vec = _mm256_mullo_epi32(a_vec, b_vec);  // Perform integer multiplication
                c_vec = _mm256_add_epi32(c_vec, mul_vec);  // Accumulate the result into c_vec
            }
            // Sum up the results in the SIMD register c_vec and store them back into C[i][j]
            int temp[8];
            _mm256_storeu_si256((__m256i*)temp, c_vec);
            C[i][j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
        }
    }
}

int main() {
  init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++) {
    auto t = get_time();
    matmul(); 
    //matmul_ikj();
    //matmul_AT();
    //matmul_BT();
    //matmul_unrolled();
    //matmul_simd();
    test();
    printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for Calculation: %f\n", avg_time / 32);
  return 0;
}

