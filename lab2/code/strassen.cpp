#include <iostream>
#include <vector>
#include<string>
#include<algorithm>
#include <random>  
#include <ctime> 
#include <sys/time.h>
#include <cassert>
#include <immintrin.h>

using namespace std;

#define I 256
#define J 512
#define K 1024
const int threshold = 32;

vector<vector<int>>A(I, vector<int>(K, 0));
vector<vector<int>>B(K, vector<int>(J, 0));
vector<vector<int>>C_groundtruth(I, vector<int>(J, 0));

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

void init() {
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < K; j++) {
            A[i][j] = rand() % 2; 
        } 
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < J; j++) {
            B[i][j] = rand()% 2; 
        }     
    }

    for(int i = 0; i < I; i++){
        for(int j = 0; j < J; j++){
            for(int k = 0; k < K; k++){
                C_groundtruth[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

vector<vector<int>>matrixMultiply(const vector<vector<int>>& X, const vector<vector<int>>& Y) {//naive
    int xRow = X.size(), xCol=X[0].size(), yCol = Y[0].size();
    vector<vector<int>>res(xRow, vector<int>(yCol, 0));
    for (int i = 0; i < xRow; i++) {
		for (int j = 0; j < yCol; j++) {
			for (int k = 0; k < xCol; k++) {
				res[i][j] += (X[i][k] * Y[k][j]);
			}
		}
	}
    return res;
}

vector<vector<int>>matrixAdd(const vector<vector<int>>& X, const vector<vector<int>>& Y) {
	int row = X.size(), col = X[0].size();
	vector<vector<int>>res(row, vector<int>(col, 0));
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			res[i][j] = X[i][j] + Y[i][j];
		}
	}
	return res;
}

vector<vector<int>>matrixSubtract(const vector<vector<int>>& X, const vector<vector<int>>& Y) {
	int row = X.size(), col = X[0].size();
	vector<vector<int>>res(row, vector<int>(col, 0));
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			res[i][j] = X[i][j] - Y[i][j];
		}
	}
	return res;
}

vector<vector<int>>Strassen(const vector<vector<int>>& X, const vector<vector<int>>& Y) {
	int xRow = X.size(), xCol = X[0].size(), yRow = Y.size(), yCol = Y[0].size();
	if (xRow == threshold) {
		return matrixMultiply(X, Y);
	}
	else {
		// X -> 4 parts
		vector<vector<int>>X00(xRow / 2, vector<int>(xCol / 2, 0));
		vector<vector<int>>X01(xRow / 2, vector<int>(xCol / 2, 0));
		vector<vector<int>>X10(xRow / 2, vector<int>(xCol / 2, 0));
		vector<vector<int>>X11(xRow / 2, vector<int>(xCol / 2, 0));
		for (int i = 0; i < xRow / 2; i++) {
			for (int j = 0; j < xCol / 2; j++) {
				X00[i][j] = X[i][j];
				X01[i][j] = X[i][j + xCol / 2];
				X10[i][j] = X[i + xRow / 2][j];
				X11[i][j] = X[i + xRow / 2][j + xCol / 2];
			}
		}

		// Y -> 4 parts
		vector<vector<int>>Y00(yRow / 2, vector<int>(yCol / 2, 0));
		vector<vector<int>>Y01(yRow / 2, vector<int>(yCol / 2, 0));
		vector<vector<int>>Y10(yRow / 2, vector<int>(yCol / 2, 0));
		vector<vector<int>>Y11(yRow / 2, vector<int>(yCol / 2, 0));
		for (int i = 0; i < yRow / 2; i++) {
			for (int j = 0; j < yCol / 2; j++) {
				Y00[i][j] = Y[i][j];
				Y01[i][j] = Y[i][j + yCol / 2];
				Y10[i][j] = Y[i + yRow / 2][j];
				Y11[i][j] = Y[i + yRow / 2][j + yCol / 2];
			}
		}
		//calculate
		vector<vector<int>>S1 = Strassen(matrixSubtract(X01, X11), matrixAdd(Y10, Y11));//(B-D)(G+H)
        vector<vector<int>>S2 = Strassen(matrixAdd(X00, X11), matrixAdd(Y00, Y11));//(A+D)(E+H)
        vector<vector<int>>S3 = Strassen(matrixSubtract(X00, X10), matrixAdd(Y00, Y01));//(A-C)(E+F)
        vector<vector<int>>S4 = Strassen(matrixAdd(X00, X01), Y11);//(A+B)H
        vector<vector<int>>S5 = Strassen(X00, matrixSubtract(Y01, Y11));//A(F-H)
		vector<vector<int>>S6 = Strassen(X11, matrixSubtract(Y10, Y00));//D(G-E)
		vector<vector<int>>S7 = Strassen(matrixAdd(X10, X11), Y00);//(C+D)E
		
        // C00 = S1 + S2 - S4 + S6, xRow * yCol
		vector<vector<int>>C00 = matrixAdd(S1, S2);
		C00 = matrixSubtract(C00, S4);
		C00 = matrixAdd(C00, S6);

        // C01 = S4 - S5, xRow * yCol
		vector<vector<int>>C01 = matrixAdd(S4, S5);

        // C10 = S6 + S7, xRow * yCol
		vector<vector<int>>C10 = matrixAdd(S6, S7);

        // C11 = S2 - S3 + S5 - S7, xRow * yCol
		vector<vector<int>>C11 = matrixSubtract(S2, S3);
		C11 = matrixAdd(C11, S5);
		C11 = matrixSubtract(C11, S7);

		//merge C
		vector<vector<int>>result(xRow, vector<int>(yCol, 0));
		for (int i = 0; i < xRow / 2; i++) {
			for (int j = 0; j < yCol / 2; j++) {
				result[i][j] = C00[i][j];
				result[i][j + yCol / 2] = C01[i][j];
				result[i + xRow / 2][j] = C10[i][j];
				result[i + xRow / 2][j + yCol / 2] = C11[i][j];
			}
		}
		return result;
	}
}

void test(vector<vector<int>>& C) {// test if the result is right
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      assert(C[i][j] == C_groundtruth[i][j]);
    }
  }
}

int main(){
    init();
    float avg_time = 0.0f;
    for (int i = 0; i < 32; i++) {
        auto t = get_time();
        
        vector<vector<int>>res  = matrixMultiply(A, B);
	    //vector<vector<int>>res = Strassen(A, B);
        test(res);
        
        printf("%f\n", get_time() - t);
        avg_time += get_time() - t;
    }
    printf("Avg Time for Calculation: %f\n", avg_time / 32);
    return 0;
}
