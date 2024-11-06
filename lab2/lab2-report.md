# CMSC5743 Lab 2

## Question 1

**Q1** Implement the matrix multiplication using Strassen Algorithm and compare the speed with original matmul() in lab 01. The shape of matrix A is I × K and the shape of matrix B is K × J. The matrix size setting remains the same as lab 01, the value of I, K, J will be fixed at 256, 512 or 1024. 

### Implement Strassen algorithm according to slide

```c++
#define I 256
#define J 512
#define K 1024

vector<vector<int>>Strassen(const vector<vector<int>>& X, const vector<vector<int>>& Y) {
	int xRow = X.size(), xCol = X[0].size(), yRow = Y.size(), yCol = Y[0].size();
	if (xRow == 1) {
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
```

### profile and compare with native matmul

| Strassen   | Native matmul |
| ---------- | ------------- |
| 2.860322 s | 0.109753 s    |

The results come from an average of 32 runs.

### Optimization

Modify the condition for entering and handling base case.

Original:

```c++
int xRow = X.size(), xCol = X[0].size(), yRow = Y.size(), yCol = Y[0].size();
	if (xRow == 1) {
		return matrixMultiply(X, Y);
	}
```

After:

```
int xRow = X.size(), xCol = X[0].size(), yRow = Y.size(), yCol = Y[0].size();
	if (xRow == threshold) {
		return matrixMultiply(X, Y);
	}
```

Profile:

| method                     | time(s)  |
| -------------------------- | -------- |
| native matmul              | 0.109753 |
| strassen (threshold = 1)   | 2.860322 |
| strassen (threshold = 2)   | 0.686267 |
| strassen (threshold = 4)   | 0.228040 |
| strassen (threshold = 8)   | 0.110396 |
| strassen (threshold = 16)  | 0.074584 |
| strassen (threshold = 32)  | 0.063404 |
| strassen (threshold = 64)  | 0.065759 |
| strassen (threshold = 128) | 0.091154 |
| strassen (threshold = 256) | 0.111582 |

Visualization:



Analysis:

> The basic problem is that you're recursing down to a leaf size of 1 with your strassen implementaiton. Strassen's algorithm has a better Big O complexity, but constants *do* matter in reality, which means in reality you're better off with a standard n^3 matrix multiplication for smaller problem sizes.
>
> So to greatly improve your program instead of doing:
>
> ```cpp
> if (tam == 1) {
>         C[0][0] = A[0][0] * B[0][0];
>         return;
>     }
> ```
>
> use `if (tam == LEAF_SIZE) // iterative solution here`. `LEAF_SIZE` should be a constant that you have to experimentally determine for your given architecture. Depending on the architecture it may be larger or smaller - there are architectures where the constant factors for strassen are so large that it's basically always worse than a simpler n^3 implementation for sensible matrix sizes. It all depends.
>
> Refer to: https://stackoverflow.com/questions/11495723/why-is-strassen-matrix-multiplication-so-much-slower-than-standard-matrix-multip

## Question 2

**Q2** Implement a C++ verison from scratch based on Winograd algorithm and compare the speed with your original im2col implement in lab 01. Please provide analysis on whether or not is your implementation improve the speed performance and why. The Convolution kernel and input size remain the same as lab 01:

- batch: 1 
- height feature: 56 
- width feature: 56 
- in channels: 3 
- out channels: 64 
- kernel size: 3 
- stride: 1 
- padding: 0



