# g++ matmul.cpp -o matmul -std=c++17 -fopt-info-optimized -mavx2 -O3 -Wall && ./matmul
g++ matmul.cpp -o matmul -std=c++17 -mavx2 -O3 -Wall && ./matmul # matmul_simd
rm -rf matmul
