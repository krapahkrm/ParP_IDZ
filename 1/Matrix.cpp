
#include "stdafx.h"
#include <Windows.h>
#include <time.h>

// Time
double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter()
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		return;

	PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}

double GetCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
}

#define N 2048

void max_of_matrix(double** matrix, int rows, int cols, double* max, int* row, int* col){
	*max = -1;
	*row = -1;
	*col = -1;
	for (int i = 0; i < rows; ++i){
		for (int j = 0; j < cols; ++j){
			if (*max >= matrix[i][j]){
				continue;
			}
			*max = matrix[i][j];
			*row = i;
			*col = j;
		}
	}
}

void matrix_mul_vector(double** matrix, double* vector, int rows, int cols, double* res){
	for (int i = 0; i < rows; ++i){
		res[i] = 0;
		for (int j = 0; j < cols; ++j){
			res[j] += matrix[i][j] * vector[j];
		}
	}
}

void matrix_mul_matrix(double** matrix1, double** matrix2, double** res, int n){
	//double r;
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; ++j){
			//r = matrix1[i][j];
			for (int k = 0; k < n; k++){
				//res[i][k] += r*matrix2[j][k];
				res[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}
}


void matrix_mul_Strassen(double** matrix1, double** matrix2, int n, double** res);
void matrix_mul(double** matrix1, double** matrix2, int startX1, int startY1, int startX2, int startY2, int n, double** res);
void matrix_add(double** matrix1, double** matrix2, int startX1, int startY1, int startX2, int startY2, int n, double** result);
void matrix_sub(double** matrix1, double** matrix2, int startX1, int startY1, int startX2, int startY2, int n, double** result);

void matrix_add(double** matrix1, double** matrix2, int startX1, int startY1, int startX2, int startY2, int n, double** result){
	for (int i = 0; i < n; ++i){
		for (int j = 0; j < n; ++j){
			result[i][j] = matrix1[startX1 + i][startY1 + j] + matrix2[startX2 + i][startY2 + j];
		}
	}
}

void matrix_sub(double** matrix1, double** matrix2, int startX1, int startY1, int startX2, int startY2, int n, double** result){
	for (int i = 0; i < n; ++i){
		for (int j = 0; j < n; ++j){
			result[i][j] = matrix1[startX1 + i][startY1 + j] - matrix2[startX2 + i][startY2 + j];
		}
	}
}

void matrix_mul_Strassen(double** matrix1, double** matrix2, int n, double** res){
	matrix_mul(matrix1, matrix2, 0, 0, 0, 0, n, res);
}

void matrix_mul(double** matrix1, double** matrix2, int sX1, int sY1, int sX2, int sY2, int n, double** res){
	int half = n / 2;
	int sX12 = sX1 + half;
	int sY12 = sY1 + half;
	int sX22 = sX2 + half;
	int sY22 = sY2 + half;

	if (n == 2){
		double p1 = (matrix1[sX1][sY1] + matrix1[sX12][sY12])
			*(matrix2[sX2][sY2] + matrix2[sX22][sY22]);
		double p2 = (matrix1[sX12][sY1] + matrix1[sX12][sY12]) *
			matrix2[sX2][sY2];
		double p3 = matrix1[sX1][sY1] *
			(matrix2[sX2][sY22] - matrix2[sX22][sY22]);
		double p4 = matrix1[sX12][sY12] *
			(matrix2[sX22][sY2] - matrix2[sX2][sY2]);
		double p5 = (matrix1[sX1][sY1] + matrix1[sX1][sY12]) *
			matrix2[sX22][sY22];
		double p6 = (matrix1[sX12][sY1] - matrix1[sX1][sY1]) *
			(matrix2[sX2][sY2] + matrix2[sX2][sY22]);
		double p7 = (matrix1[sX1][sY12] - matrix1[sX12][sY12])*
			(matrix2[sX22][sY2] + matrix2[sX22][sY22]);

		res[0][0] = p1 + p4 - p5 + p7;
		res[0][1] = p3 + p5;
		res[1][0] = p2 + p4;
		res[1][1] = p1 - p2 + p3 + p6;
	}
	else{
		for (int i = 0; i < half; ++i){
			for (int j = 0; j < half; ++j){
				res[i][j] = 0;
				res[i + half][j] = 0;
				res[i + half][j + half] = 0;
			}
		}

		double** p1 = new double*[half];
		double** p2 = new double*[half];
		for (int i = 0; i < half; ++i){
			p1[i] = new double[half];
			p2[i] = new double[half];
		}

		// p1
		for (int i = 0; i < half; ++i){
			for (int j = 0; j < half; ++j){
				p1[i][j] = matrix1[sX1 + i][sY1 + j] + matrix1[sX12 + i][sY12 + j];
				res[i][j + half] = matrix2[sX2 + i][sY2 + j] + matrix2[sX22 + i][sY22 + j];
			}
		}
		matrix_mul(p1, res, 0, 0, 0, half, half, p2);
		for (int i = 0; i < half; ++i){
			for (int j = 0; j < half; ++j){
				// c11
				res[i][j] += p2[i][j];
				// c22
				res[i + half][j + half] += p2[i][j];
			}
		}

		// p6
		for (int i = 0; i < half; ++i){
			for (int j = 0; j < half; ++j){
				p1[i][j] = matrix1[sX12 + i][sY1 + j] - matrix1[sX1 + i][sY1 + j];
				res[i][j + half] = matrix2[sX2 + i][sY2 + j] + matrix2[sX2 + i][sY22 + j];
			}
		}
		matrix_mul(p1, res, 0, 0, 0, half, half, p2);
		for (int i = 0; i < half; ++i){
			for (int j = 0; j < half; ++j){
				// c22
				res[i + half][j + half] += p2[i][j];
			}
		}

		// p7
		for (int i = 0; i < half; ++i){
			for (int j = 0; j < half; ++j){
				p1[i][j] = matrix1[sX1 + i][sY12 + j] - matrix1[sX12 + i][sY12 + j];
				res[i][j + half] = matrix2[sX22 + i][sY2 + j] + matrix2[sX22 + i][sY22 + j];
			}
		}
		matrix_mul(p1, res, 0, 0, 0, half, half, p2);
		for (int i = 0; i < half; ++i){
			for (int j = 0; j < half; ++j){
				// c11
				res[i][j] += p2[i][j];
			}
		}

		// p2
		matrix_add(matrix1, matrix1, sX12, sY1, sX12, sY12, half, p1);
		matrix_mul(p1, matrix2, 0, 0, sX2, sY2, half, p2);
		for (int i = 0; i < half; ++i){
			for (int j = 0; j < half; ++j){
				// c21
				res[i + half][j] += p2[i][j];
				// c22
				res[i + half][j + half] -= p2[i][j];
			}
		}

		// p4
		matrix_sub(matrix2, matrix2, sX22, sY2, sX2, sY2, half, p1);
		matrix_mul(matrix1, p1, sX12, sY12, 0, 0, half, p2);
		for (int i = 0; i < half; ++i){
			for (int j = 0; j < half; ++j){
				// c11
				res[i][j] += p2[i][j];
				// c21
				res[i + half][j] += p2[i][j];
			}
		}
		for (int i = 0; i < half; ++i){
			for (int j = 0; j < half; ++j){
				res[i][j + half] = 0;
			}
		}

		// p3
		matrix_sub(matrix2, matrix2, sX2, sY22, sX22, sY22, half, p1);
		matrix_mul(matrix1, p1, sX1, sY1, 0, 0, half, p2);
		for (int i = 0; i < half; ++i){
			for (int j = 0; j < half; ++j){
				// c12
				res[i][j + half] += p2[i][j];
				// c22
				res[i + half][j + half] += p2[i][j];
			}
		}

		// p5
		matrix_add(matrix1, matrix1, sX1, sY1, sX1, sY12, half, p1);
		matrix_mul(p1, matrix2, 0, 0, sX22, sY22, half, p2);
		for (int i = 0; i < half; ++i){
			for (int j = 0; j < half; ++j){
				// c11
				res[i][j] -= p2[i][j];
				// c12
				res[i][j + half] += p2[i][j];
			}
		}

		for (int i = 0; i < half; ++i){
			delete p1[i];
			delete p2[i];
		}
		delete[] p1;
		delete[] p2;
	}
}

bool matrix_compare(double** matrix1, double** matrix2, int n){
	for (int i = 0; i < n; ++i){
		for (int j = 0; j < n; ++j){
			if (matrix1[i][j] == matrix2[i][j]){
				continue;
			}
			_tcprintf(_T("%f %f\n"), matrix1[i][j], matrix2[i][j]);
			return false;
		}
	}
	return true;
}

void test_for_matrix(int n){

	double** matrixA = new double*[n];
	double** matrixB = new double*[n];
	double** res1 = new double*[n];
	double** res2 = new double*[n];

	for (int i = 0; i < n; ++i){
		matrixA[i] = new double[n];
		matrixB[i] = new double[n];
		res2[i] = new double[n];
		res1[i] = new double[n];
		for (int j = 0; j < n; ++j){
			matrixA[i][j] = (double)(rand() % 100 + 1);
			matrixB[i][j] = (double)(rand() % 100 + 1);
			res2[i][j] = 0;
		}
	}


	StartCounter();
	matrix_mul_matrix(matrixA, matrixB, res2, n);
	float time_end = GetCounter();

	_tcprintf(_T("Multiple by n^3 for %d x %d :              %f ms\n"), n,n,time_end);

	StartCounter();
	matrix_mul_Strassen(matrixA, matrixB, n, res1);
	time_end = GetCounter();

	_tcprintf(_T("Multiple by Strassen (n^2.81) for %d x %d: %f ms\n\n"), n, n, time_end);

	if (!matrix_compare(res1, res2, n)){
		_tcprintf(_T("Matrix A ! = Matrix B\n\n"));
	}

}



int _tmain(int argc, _TCHAR* argv[])
{
	srand(time(0));
	/*

	int arr[6] = { 64, 128, 256, 512, 1024, 2048 };
	// Initilize
	for (int k = 0; k < 6; k++)
	{
		double** matrix = new double*[arr[k]];
		double* vector = new double[arr[k]];
		double* vectorRes = new double[arr[k]];

		for (int i = 0; i < arr[k]; ++i){
			matrix[i] = new double[arr[k]];
			vector[i] = (double)rand();
			for (int j = 0; j < arr[k]; ++j){
				matrix[i][j] = (double)rand();
			}
		}

		// Max of matrix
		
		double max;
		int row;
		int col;

		StartCounter();
		max_of_matrix(matrix, arr[k], arr[k], &max, &row, &col);
		float time_end = GetCounter();
		_tcprintf(_T("Max value = %.1f [%d,%d]\n"), max, row, col);
		_tcprintf(_T("Time for %d x %d: %f ms\n\n"), arr[k], arr[k], time_end);

		
		// Matrix mul Vector
		StartCounter();
		matrix_mul_vector(matrix, vector, arr[k], arr[k], vectorRes);
		float time_end = GetCounter();
		_tcprintf(_T("Multiple matrix to vector for %d x %d. Time : %f ms\n\n"), arr[k], arr[k], time_end);
	}
*/
	// Matrix mul Matrix

	test_for_matrix(64);
	test_for_matrix(128);
	test_for_matrix(256);
	test_for_matrix(512);
	test_for_matrix(1024);
	test_for_matrix(2048);

	return 0;
}

