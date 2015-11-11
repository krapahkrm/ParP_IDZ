// SIMD.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <time.h>
#include <Windows.h>
#include <intrin.h>
#include <math.h>
//#include <algorithm>

#define N 256

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

bool areEqual(double x, double y)
{
	static double SMALL_NUM = 1.0E-5;
	if (abs(x) < SMALL_NUM && abs(y) < SMALL_NUM)
		return abs(x - y) < SMALL_NUM;
	else
		return abs(x - y) < abs(x) * SMALL_NUM;
}

void max_of_matrix(double* matrix, int rows, int cols, double* max, int* row, int* col){
	*max = -1;
	*row = -1;
	*col = -1;
	for (int i = 0; i < rows; ++i){
		for (int j = 0; j < cols; ++j){
			if (*max >= matrix[i*rows + j]){
				continue;
			}
			*max = matrix[i*rows + j];
			*row = i;
			*col = j;
		}
	}
}

void max_of_matrix_only_max(double* matrix, int rows, int cols, double* max){
	*max = -1;
	for (int i = 0; i < rows; ++i){
		for (int j = 0; j < cols; ++j){
			if (*max >= matrix[i*rows + j]){
				continue;
			}
			*max = matrix[i*rows + j];
		}
	}
}

void simd_max_of_matrix(double* matrix, int rows, int cols, double* max, int* row, int* col)
{
	__m128d *pa = (__m128d *)matrix;
	__m128d m = pa[0];
	__m128d temp;
	int len = rows*cols / 2;
	int index1=0,index2=0;
	for (int i = 1; i < len; i++)
	{
		m = _mm_max_pd(m, pa[i]);
		if (m.m128d_f64[0] == pa[i].m128d_f64[0])
		{
			index1 = i;
		}
		if (m.m128d_f64[1] == pa[i].m128d_f64[1])
		{
			index2 = i;
		}
	}
	
	if (m.m128d_f64[1] > m.m128d_f64[0]){
		*max = m.m128d_f64[1];
		*row = (int)(index2 / (rows/2));
		*col = (index2*2)%cols;
		*col += 1;
	}
	else
	{
		*row = (int)(index1 / (rows/2));
		*col = (index1*2)%cols;
		*max = m.m128d_f64[0];
	}
	//*max = m.m128d_f64[0]>m.m128d_f64[1] ? m.m128d_f64[0] : m.m128d_f64[1];
}

void simd_max_of_matrix_only_max(double* matrix, int rows, int cols, double* max)
{
	__m128d *pa = (__m128d *)matrix;
	__m128d m = pa[0];
	int len = rows*cols/2;
	for (int i = 1; i < len; i++)
	{
		m = _mm_max_pd(m, pa[i]);
	}
	*max = m.m128d_f64[0]>m.m128d_f64[1] ? m.m128d_f64[0] : m.m128d_f64[1];
}


void test_max()
{
	int R[] = {64,128,256,512,1024,2048};
	for (int r = 0; r < 6; r++)
	{
		int len = R[r];
		double* matrix = (double*)_aligned_malloc(sizeof(double)*len*len, 16);
		for (int i = 0; i < len; ++i){
			for (int j = 0; j < len; ++j){
				matrix[i*len + j] = (double)(rand() % 100);
			}
		}
		matrix[len * len / 2 + len / 2] = 100.0; // maximum

		double max;
		int row;
		int col;

		_tcprintf(_T("-------------- %d ---------------\n"), len);

		StartCounter();
		max_of_matrix(matrix, len, len, &max, &row, &col);
		float time_end = GetCounter();
		_tcprintf(_T("Max value = %.1f [%d,%d]\n"), max, row, col);
		_tcprintf(_T("Time for %d x %d: %f ms\n\n"), len, len, time_end);

		StartCounter();
		simd_max_of_matrix(matrix, len, len, &max, &row, &col);
		time_end = GetCounter();
		_tcprintf(_T("Max value = %.1f [%d,%d]\n"), max, row, col);
		_tcprintf(_T("Time SIMD for %d x %d: %f ms\n"), len, len, time_end);

		StartCounter();
		simd_max_of_matrix_only_max(matrix, len, len, &max);
		time_end = GetCounter();
		_tcprintf(_T("Max value = %.1f\n"), max);
		_tcprintf(_T("Time SIMD only max for %d x %d: %f ms\n"), len, len, time_end);
		
	}
	
}

void test_max_only()
{
	int R[] = { 64, 128, 256, 512, 1024, 2048, 4096,8192};
	for (int r = 0; r < 8; r++)
	{
		int len = R[r];
		double* matrix = (double*)_aligned_malloc(sizeof(double)*len*len, 16);
		for (int i = 0; i < len; ++i){
			for (int j = 0; j < len; ++j){
				matrix[i*len + j] = (double)(rand() % 100);
			}
		}
		matrix[len * len / 2 + len / 2] = 100.0; // maximum

		double max;
		int row;
		int col;

		_tcprintf(_T("-------------- %d ---------------\n"), len);

		StartCounter();
		max_of_matrix_only_max(matrix, len, len, &max);
		float time_end = GetCounter();
		_tcprintf(_T("Max value = %.1f\n"), max);
		_tcprintf(_T("Time for %d x %d: %f ms\n\n"), len, len, time_end);

		StartCounter();
		simd_max_of_matrix_only_max(matrix, len, len, &max);
		time_end = GetCounter();
		_tcprintf(_T("Max value = %.1f\n"), max);
		_tcprintf(_T("Time SIMD only max for %d x %d: %f ms\n"), len, len, time_end);

	}

}

void print_vector(double* v, int rows)
{
	for (int i = 0; i <rows; i++)
	{

		_tcprintf(_T("%.0lf "), v[i]);

	}
	_tcprintf(_T("\n"));
}

void print_matrix(double* v, int rows, int cols)
{
	for (int i = 0; i <rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			_tcprintf(_T("%.0lf "), v[i*rows + j]);
		}
		_tcprintf(_T("\n"));

	}
	_tcprintf(_T("----\n"));
}

void matrix_mul_vector(double* matrix, double* vector, int rows, int cols, double* res){
	for (int i = 0; i < rows; i++)
	{
		res[i] = 0;
	}
	for (int i = 0; i < rows; ++i){
		
		for (int j = 0; j < cols; ++j){
			res[j] += matrix[i*rows+j] * vector[j];
		}
	}
}

void matrix_mul_vector_simd(double* matrix, double* vector, int rows, int cols, double* res){
	__m128d *m = (__m128d*)matrix, *v = (__m128d*)vector, *r = (__m128d*)res;
	for (int j = 0; j < cols / 2; j++)
	{
		r[j]=_mm_setr_pd(0,0);
	}

	int len = rows*cols / 2;
	for (int i = 0; i < len; i+=cols/2){
		for (int j = 0; j < cols/2; ++j){
			r[j] = _mm_add_pd(r[j], _mm_mul_pd(m[i + j], v[j]));
		}
	}
}

int compare_vector(double* v1, double* v2, int rows)
{
	int count = 0;
	int len = rows;
	for (int i = 0; i <rows; i++)
	{
		if (!areEqual(v1[i],v2[i])) count++;
	}
	return count;
}

int compare_matrix(double* v1, double* v2, int rows)
{
	int count = 0;
	int len = rows*rows;
	for (int i = 0; i <len; i++)
	{
		if (!areEqual(v1[i], v2[i])) count++;
	}
	return count;
}

void test_mul_vector()
{
	int R[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
	for (int r = 0; r < 8; r++)
	{
		int len = R[r];
		double* matrix = (double*)_aligned_malloc(sizeof(double)* len * len, 16);
		double* vector = (double*)_aligned_malloc(sizeof(double)* len, 16);
		double* res = (double*)_aligned_malloc(sizeof(double)* len, 16);
		double* res2 = (double*)_aligned_malloc(sizeof(double)* len, 16);
		for (int i = 0; i < len; ++i){
			vector[i] = (double)(rand() % 3);
			for (int j = 0; j < len; ++j){
				matrix[i * len + j] = (double)(rand() % 3);
			}
		}

		_tcprintf(_T("-------------- %d ---------------\n"), len);
		StartCounter();
		matrix_mul_vector(matrix, vector, len, len, res);
		float time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d on vector %d: %f ms\n"), len, len, len, time_end);

		StartCounter();
		matrix_mul_vector_simd(matrix, vector, len, len, res2);
		time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d on vector %d: %f ms\n"), len, len, len, time_end);

		_tcprintf(_T("Count of different elements: %d\n\n"), compare_vector(res, res2, len));
	}
}

void matrix_mul_matrix(double* matrix1, double* matrix2, double* res, int n){
	double r;
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; ++j){
			r = matrix1[i*n + j];
			for (int k = 0; k < n; k++){
				res[i*n+k] += r*matrix2[j*n+k];
				//res[i*n+j] += matrix1[i*n+k] * matrix2[k*n+j];
			}
		}
	}
}

void matrix_mul_matrix_simd(double* matrix1, double* matrix2, double* res, int n){
	__m128d *m1 = (__m128d*)matrix1, *m2 = (__m128d*)matrix2, *r = (__m128d*)res;
	__m128d temp;
	int len = n*n / 2;
	for (int i = 0; i < len; i+=n/2){
		for (int j = 0; j < n; j++){
			temp = _mm_setr_pd(0, 0);
			for (int k = i; k < i + n / 2; i++)
			{
				temp = _mm_add_pd(temp, _mm_mul_pd(m1[k], m2[k+j*n/2]));
			}
			res[i/2+j] = temp.m128d_f64[0] + temp.m128d_f64[1];
		}
	}
}

void test_mul_matrix()
{
	int R[] = { 64, 128, 256, 512, 1024, 2048};
	for (int r = 0; r < 6; r++)
	{
		int len = R[r];
		double* matrix1 = (double*)_aligned_malloc(sizeof(double)* len * len, 16);
		double* matrix2 = (double*)_aligned_malloc(sizeof(double)* len* len, 16);
		double* res = (double*)_aligned_malloc(sizeof(double)* len*len, 16);
		double* res2 = (double*)_aligned_malloc(sizeof(double)* len*len, 16);

		for (int i = 0; i < len; ++i){
			for (int j = 0; j < len; ++j){
				matrix1[i * len + j] = (double)(rand() % 3);
				matrix2[i * len + j] = (double)(rand() % 3);
				res[i*len + j] = 0.0;
				res2[i*len + j] = 0.0;
			}
		}

		_tcprintf(_T("-------------- %d ---------------\n"), len);
		StartCounter();
		matrix_mul_matrix(matrix1, matrix2, res, len);
		float time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d: %f ms\n"), len, len, time_end);

		StartCounter();
		matrix_mul_matrix(matrix1, matrix2, res2, len);
		time_end = GetCounter();
		_tcprintf(_T("Time SIMD for multiple matrix %d x %d: %f ms\n"), len, len, time_end);

		_tcprintf(_T("Count of different elements: %d\n\n"), compare_matrix(res, res2, len));
	}
}
int _tmain(int argc, _TCHAR* argv[])
{
	srand(time(0));
	//test_max();
	//test_max_only();
	//test_mul_vector();
	test_mul_matrix();
	return 0;
}

