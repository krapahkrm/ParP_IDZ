// SIMD.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <time.h>
#include <Windows.h>
#include <intrin.h>
#include <math.h>

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
	static double SMALL_NUM = 1.0E-7;
	if (abs(x) < SMALL_NUM && abs(y) < SMALL_NUM)
		return abs(x - y) < SMALL_NUM;
	else
		return abs(x - y) < abs(x) * SMALL_NUM;
}

void avx_support(){
	int cpuInfo[4];
	__cpuidex(cpuInfo, 1, 0);
	bool avxSupported = cpuInfo[2] & 1 << 28;
	if (avxSupported)
	{
		printf("AVX is supported by __cpuid\n");
	}
	else
	{
		printf("AVX is NOT supported by __cpuid\n");
	}
#ifdef __AVX__
	printf("AVX is supported by DEBUG\n");
#else
	printf("AVX is NOT supported by DEBUG\n");
#endif
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


void sse_max_of_matrix(double* matrix, int rows, int cols, double* max, int* row, int* col)
{
	__m128d *pa = (__m128d *)matrix, m = pa[0];
	int len = rows*cols / 2;
	for (int i = 1; i < len; i++)
	{
		m = _mm_max_pd(m, pa[i]);
	}

	*max = m.m128d_f64[0];
	if (m.m128d_f64[1]>*max)
	{
		*max = m.m128d_f64[1];
	}
	len /= 2;
	for (int i = 0; i < len; i++)
	{
		if (matrix[i]== *max)
		{
			*row = i / rows;
			*col = i % cols;
			break;
		}
	}
}


void avx_max_of_matrix(double* matrix, int rows, int cols, double* max, int* row, int* col)
{
	__m256d *pa = (__m256d *)matrix, m = pa[0];
	int len = rows*cols / 4;
	for (int i = 1; i < len; i++)
	{
		m = _mm256_max_pd(m, pa[i]);
	}

	*max = m.m256d_f64[0];
	for (int i = 1; i < 4; i++)
	{
		if (*max < m.m256d_f64[i])
		{
			*max = m.m256d_f64[i];
		}
	}
	len /= 4;
	for (int i = 0; i < len; i++)
	{
		if (matrix[i] == *max)
		{
			*row = i / rows;
			*col = i % cols;
			break;
		}
	}
}



void test_max()
{
	int R[] = {64,128,256,512,1024,2048,4096,8192};
	for (int r = 0; r < 8; r++)
	{
		int len = R[r];
		double* matrix = (double*)_aligned_malloc(sizeof(double)*len*len, 16);
		for (int i = 0; i < len; ++i){
			for (int j = 0; j < len; ++j){
				matrix[i*len + j] = (double)(rand() % 100);
			}
		}

		double max;
		int row;
		int col;

		_tcprintf(_T("---------------------------- %d x %d -----------------------------\n"), len,len);

		StartCounter();
		max_of_matrix(matrix, len, len, &max, &row, &col);
		double time_end = GetCounter();
		_tcprintf(_T("Max value = %.1f [%d,%d]    \t"), max, row, col);
		_tcprintf(_T("Time:    \t %f ms\n\n"),time_end);

		StartCounter();
		sse_max_of_matrix(matrix, len, len, &max, &row, &col);
		time_end = GetCounter();
		_tcprintf(_T("Max value = %.1f [%d,%d]    \t"), max, row, col);
		_tcprintf(_T("Time SSE:\t %f ms\n\n"),  time_end);

#ifdef __AVX__	
			StartCounter();
			avx_max_of_matrix(matrix, len, len, &max, &row, &col);
			time_end = GetCounter();
			_tcprintf(_T("Max value = %.1f [%d,%d]    \t"), max, row, col);
			_tcprintf(_T("Time AVX:\t %f ms\n"), time_end);
#endif
		_aligned_free(matrix);
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

void matrix_mul_vector_sse(double* matrix, double* vector, int rows, int cols, double* res){
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

void matrix_mul_vector_avx(double* matrix, double* vector, int rows, int cols, double* res){
	__m256d *m = (__m256d*)matrix, *v = (__m256d*)vector, *r = (__m256d*)res;
	for (int j = 0; j < cols / 4; j++)
	{
		r[j] = _mm256_setzero_pd();
	}

	int len = rows*cols / 4;
	for (int i = 0; i < len; i += cols / 4){
		for (int j = 0; j < cols / 4; ++j){
			r[j] = _mm256_add_pd(r[j], _mm256_mul_pd(m[i + j], v[j]));
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
		double time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d on vector %d: %f ms\n"), len, len, len, time_end);

		StartCounter();
		matrix_mul_vector_sse(matrix, vector, len, len, res2);
		time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d on vector %d: %f ms (SSE)\n"), len, len, len, time_end);

		_tcprintf(_T("--Count of different elements: %d\n"), compare_vector(res, res2, len));

#ifdef __AVX__	
		double* res3 = (double*)_aligned_malloc(sizeof(double)* len, 16);
		StartCounter();
		matrix_mul_vector_avx(matrix, vector, len, len, res3);
		time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d on vector %d: %f ms (AVX)\n"), len, len, len, time_end);

		_tcprintf(_T("--Count of different elements: %d\n"), compare_vector(res, res3, len));
		_aligned_free(res3);
#endif
		_aligned_free(matrix);
		_aligned_free(vector);
		_aligned_free(res);
		_aligned_free(res2);
		
	}
}

void matrix_mul_matrix(double* matrix1, double* matrix2, double* res, int n){
	double r;
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; ++j){
			r = matrix1[i*n + j];
			for (int k = 0; k < n; k++){
				res[i*n+k] += r*matrix2[j*n+k];
			}
		}
	}
}

void matrix_mul_matrix_sse(double* matrix1, double* matrix2, double* res, int n){
	__m128d *m2 = (__m128d*)matrix2, *arr = (__m128d*)res;
	__m128d temp;
	double r;
	int len = n / 2;
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; ++j){
			r = matrix1[i*n + j];
			for (int k = 0; k < len; k++){
				temp = _mm_set1_pd(r);
				arr[i * len + k] = _mm_add_pd(arr[i* len + k],
					_mm_mul_pd(temp, m2[j * len + k]));
			}
		}
	}
}

void matrix_mul_matrix_avx(double* matrix1, double* matrix2, double* res, int n){
	__m256d *m2 = (__m256d*)matrix2, *arr = (__m256d*)res;
	__m256d temp;
	double r;
	int len = n / 4;
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; ++j){
			r = matrix1[i*n + j];
			for (int k = 0; k < len; k++){
				temp = _mm256_set1_pd(r);
				arr[i*len + k] = _mm256_add_pd(arr[i*len + k],
					_mm256_mul_pd(temp, m2[j*len + k]));
			}
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
		double time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d: %f ms\n"), len, len, time_end);

		StartCounter();
		matrix_mul_matrix_sse(matrix1, matrix2, res2, len);
		time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d: %f ms (SSE)\n"), len, len, time_end);

		_tcprintf(_T("--Count of different elements: %d\n"), compare_matrix(res, res2, len));


#ifdef __AVX__	
		double* res3 = (double*)_aligned_malloc(sizeof(double)* len*len, 16);
		for (int i = 0; i < len; ++i){
			for (int j = 0; j < len; ++j){
				res3[i*len+j]=0.0;
			}}
		StartCounter();
		matrix_mul_matrix_avx(matrix1, matrix2, res3, len);
		time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d: %f ms (AVX)\n"), len, len, time_end);

		_tcprintf(_T("--Count of different elements: %d\n"), compare_matrix(res, res3, len));
		_aligned_free(res3);
#endif


		_aligned_free(matrix1);
		_aligned_free(matrix2);
		_aligned_free(res);
		_aligned_free(res2);
	}
}
int _tmain(int argc, _TCHAR* argv[])
{
	srand((unsigned int)time(0));
	avx_support();
	//test_max();
	//test_mul_vector();
	test_mul_matrix();
	return 0;
}

