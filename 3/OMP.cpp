// OMP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <Windows.h>
#include <math.h>
#include <intrin.h>
#include <omp.h>
#include <time.h>

#pragma region Settings
bool avxSupported;
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

bool areEqualMany(double x, double y)
{
	static double SMALL_NUM = 1.0E-2;
	if (abs(x - y) / max(x, y) < SMALL_NUM) return true;
	return false;
}
void avx_support(){
	int cpuInfo[4];
	__cpuidex(cpuInfo, 1, 0);
	avxSupported = cpuInfo[2] & 1 << 28;
	if (avxSupported)
	{
		printf("AVX is supported by __cpuid\n");
	}
	else
	{
		printf("AVX is NOT supported by __cpuid\n");
	}

}
#pragma endregion

#pragma region MAX

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

void omp_max_of_matrix(double* matrix, int rows, int cols, double* max, int* row, int* col){
	*max = -1;
	*row = -1;
	*col = -1;
	double _max = -1;
	int _row = -1, _col = -1;
#pragma omp parallel for shared(matrix) private(_max,_col,_row)
	for (int i = 0; i < rows; ++i){
		_max = -1;
		for (int j = 0; j < cols; ++j){
			if (_max >= matrix[i*rows + j]){
				continue;
			}
			_max = matrix[i*rows + j];
			_row = i;
			_col = j;
		}
#pragma omp critical
		{
			if (*max < _max || ((*row>_row || (*row == _row && *col>_col)) && areEqual(*max, _max))){
				*max = _max;
				*row = _row;
				*col = _col;
			}

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
		if (matrix[i] == *max)
		{
			*row = i / rows;
			*col = i % cols;
			break;
		}
	}
}

void omp_sse_max_of_matrix(double* matrix, int rows, int cols, double* max, int* row, int* col)
{
	__m128d *pa = (__m128d *)matrix, m = pa[0];
	int len = rows*cols / 2;
#pragma omp parallel for
	for (int i = 0; i < rows / 2; i++)
	{
		__m128d loc_m = _mm_setzero_pd();
		for (int j = 0; j < cols; j++)
		{
			loc_m = _mm_max_pd(loc_m, pa[i*rows + j]);
		}

#pragma omp critical
		{
			m = _mm_max_pd(m, loc_m);
		}
	}

	*max = m.m128d_f64[0];
	if (m.m128d_f64[1]>*max)
	{
		*max = m.m128d_f64[1];
	}
	len /= 2;
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

void omp_avx_max_of_matrix(double* matrix, int rows, int cols, double* max, int* row, int* col)
{
	__m256d *pa = (__m256d *)matrix, m = pa[0];
	int len = rows*cols / 4;
#pragma omp parallel for
	for (int i = 0; i < rows / 4; i++)
	{
		__m256d loc_m = _mm256_setzero_pd();
		for (int j = 0; j < cols; j++)
		{
			loc_m = _mm256_max_pd(loc_m, pa[i*rows + j]);
		}
#pragma omp critical
		{
			m = _mm256_max_pd(m, loc_m);
		}
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
	int R[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
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

		_tcprintf(_T("---------------------------- %d x %d -----------------------------\n"), len, len);

		StartCounter();
		max_of_matrix(matrix, len, len, &max, &row, &col);
		double time_end = GetCounter();
		_tcprintf(_T("Max value = %.1f [%d,%d]    \t"), max, row, col);
		_tcprintf(_T("Time:    \t %f ms\n\n"), time_end);

		StartCounter();
		omp_max_of_matrix(matrix, len, len, &max, &row, &col);
		time_end = GetCounter();
		_tcprintf(_T("Max value = %.1f [%d,%d]    \t"), max, row, col);
		_tcprintf(_T("Time OMP:    \t %f ms\n\n"), time_end);


		StartCounter();
		sse_max_of_matrix(matrix, len, len, &max, &row, &col);
		time_end = GetCounter();
		_tcprintf(_T("Max value = %.1f [%d,%d]    \t"), max, row, col);
		_tcprintf(_T("Time SSE:\t %f ms\n\n"), time_end);

		StartCounter();
		omp_sse_max_of_matrix(matrix, len, len, &max, &row, &col);
		time_end = GetCounter();
		_tcprintf(_T("Max value = %.1f [%d,%d]    \t"), max, row, col);
		_tcprintf(_T("Time SSE OMP:\t %f ms\n\n"), time_end);

		if (avxSupported)
		{
			StartCounter();
			avx_max_of_matrix(matrix, len, len, &max, &row, &col);
			time_end = GetCounter();
			_tcprintf(_T("Max value = %.1f [%d,%d]    \t"), max, row, col);
			_tcprintf(_T("Time AVX:\t %f ms\n\n"), time_end);

			StartCounter();
			omp_avx_max_of_matrix(matrix, len, len, &max, &row, &col);
			time_end = GetCounter();
			_tcprintf(_T("Max value = %.1f [%d,%d]    \t"), max, row, col);
			_tcprintf(_T("Time AVX OMP:\t %f ms\n\n"), time_end);

		}

		_aligned_free(matrix);

	}

}

#pragma endregion

#pragma region Matrix*Vector

void matrix_mul_vector(double* matrix, double* vector, int rows, int cols, double* res){
	for (int i = 0; i < rows; i++)
	{
		res[i] = 0;
	}
	for (int i = 0; i < rows; ++i){

		for (int j = 0; j < cols; ++j){
			res[j] += matrix[i*rows + j] * vector[j];
		}
	}
}

void omp_matrix_mul_vector(double* matrix, double* vector, int rows, int cols, double* res){
#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		res[i] = 0;
	}
	int i = 0, j = 0;
#pragma omp parallel for shared(res,matrix,vector) private(i,j)
	for (i = 0; i < rows; ++i){
		for (j = 0; j < cols; ++j){
			res[j] += matrix[i*rows + j] * vector[j];
		}
	}
}

void matrix_mul_vector_sse(double* matrix, double* vector, int rows, int cols, double* res){
	__m128d *m = (__m128d*)matrix, *v = (__m128d*)vector, *r = (__m128d*)res;
	for (int j = 0; j < cols / 2; j++)
	{
		r[j] = _mm_setr_pd(0, 0);
	}

	int len = rows*cols / 2;
	for (int i = 0; i < len; i += cols / 2){
		for (int j = 0; j < cols / 2; ++j){
			r[j] = _mm_add_pd(r[j], _mm_mul_pd(m[i + j], v[j]));
		}
	}
}

void omp_matrix_mul_vector_sse(double* matrix, double* vector, int rows, int cols, double* res){
	__m128d *m = (__m128d*)matrix, *v = (__m128d*)vector, *r = (__m128d*)res;
#pragma omp parallel for
	for (int j = 0; j < cols / 2; j++)
	{
		r[j] = _mm_setr_pd(0, 0);
	}

	int len = rows*cols / 2;
	int i = 0, j = 0;
#pragma omp parallel for shared(r,m,v) private(i,j)
	for (i = 0; i < len; i += cols / 2){
		for (j = 0; j < cols / 2; ++j){
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

void omp_matrix_mul_vector_avx(double* matrix, double* vector, int rows, int cols, double* res){
	__m256d *m = (__m256d*)matrix, *v = (__m256d*)vector, *r = (__m256d*)res;
#pragma omp parallel for
	for (int j = 0; j < cols / 4; j++)
	{
		r[j] = _mm256_setzero_pd();
	}

	int len = rows*cols / 4;
	int i = 0, j = 0;
#pragma omp parallel for shared(r,m,v) private(i,j)
	for (i = 0; i < len; i += cols / 4){
		for (j = 0; j < cols / 4; ++j){
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
		if (!areEqualMany(v1[i], v2[i])) count++;
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
		double* res_omp = (double*)_aligned_malloc(sizeof(double)* len, 16);
		double* res_sse= (double*)_aligned_malloc(sizeof(double)* len, 16);
		double* res_sse_omp = (double*)_aligned_malloc(sizeof(double)* len, 16);

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
		omp_matrix_mul_vector(matrix, vector, len, len, res_omp);
		time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d on vector %d: %f ms (OMP)\n"), len, len, len, time_end);

		_tcprintf(_T("--Count of different elements: %d\n"), compare_vector(res, res_omp, len));

		StartCounter();
		matrix_mul_vector_sse(matrix, vector, len, len, res_sse);
		time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d on vector %d: %f ms (SSE)\n"), len, len, len, time_end);

		StartCounter();
		omp_matrix_mul_vector_sse(matrix, vector, len, len, res_sse_omp);
		time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d on vector %d: %f ms (SSE OMP)\n"), len, len, len, time_end);

		_tcprintf(_T("--Count of different elements: %d\n"), compare_vector(res_sse, res_sse_omp, len));

		if (avxSupported)
		{
			double* res_avx = (double*)_aligned_malloc(sizeof(double)* len, 16);
			double* res_avx_omp = (double*)_aligned_malloc(sizeof(double)* len, 16);
			StartCounter();
			matrix_mul_vector_avx(matrix, vector, len, len, res_avx);
			time_end = GetCounter();
			_tcprintf(_T("Time for multiple matrix %d x %d on vector %d: %f ms (AVX)\n"), len, len, len, time_end);

			StartCounter();
			omp_matrix_mul_vector_avx(matrix, vector, len, len, res_avx_omp);
			time_end = GetCounter();
			_tcprintf(_T("Time for multiple matrix %d x %d on vector %d: %f ms (AVX OMP)\n"), len, len, len, time_end);

			_tcprintf(_T("--Count of different elements: %d\n"), compare_vector(res_avx, res_avx_omp, len));
			_aligned_free(res_avx);
			_aligned_free(res_avx_omp);
		}


		_aligned_free(matrix);
		_aligned_free(vector);
		_aligned_free(res);
		_aligned_free(res_omp);
		_aligned_free(res_sse_omp);
		_aligned_free(res_sse);

	}
}

#pragma endregion

#pragma region Matrix*Matrix

int compare_matrix(double* v1, double* v2, int rows)
{
	int count = 0;
	int len = rows*rows;
	for (int i = 0; i <len; i++)
	{
		if (!areEqualMany(v1[i], v2[i])) count++;
	}
	return count;
}

void matrix_mul_matrix(double* matrix1, double* matrix2, double* res, int n){
	double r;
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; ++j){
			r = matrix1[i*n + j];
			for (int k = 0; k < n; k++){
				res[i*n + k] += r*matrix2[j*n + k];
			}
		}
	}
}

void omp_matrix_mul_matrix(double* matrix1, double* matrix2, double* res, int n){
	double r=0.0;
	int i = 0, j = 0, k = 0;
#pragma omp parallel for shared(matrix1,matrix2,res) private(i,j,k,r)
	for (i = 0; i < n; i++){
		for (j = 0; j < n; ++j){
			r = matrix1[i*n + j];
			for (k = 0; k < n; k++){
				res[i*n + k] += r*matrix2[j*n + k];
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

void omp_matrix_mul_matrix_sse(double* matrix1, double* matrix2, double* res, int n){
	__m128d *m2 = (__m128d*)matrix2, *arr = (__m128d*)res;
	__m128d temp = _mm_setzero_pd();
	double r=0.0;
	int len = n / 2;
	int i = 0, j = 0, k = 0;
#pragma omp parallel for shared(matrix1,matrix2,res) private(i,j,k,r,temp)
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

void omp_matrix_mul_matrix_avx(double* matrix1, double* matrix2, double* res, int n){
	__m256d *m2 = (__m256d*)matrix2, *arr = (__m256d*)res;
	__m256d temp=_mm256_setzero_pd();
	double r=0.0;
	int len = n / 4;
	int i = 0, j = 0, k = 0;
#pragma omp parallel for shared(matrix1,matrix2,res) private(i,j,k,temp)
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
	int R[] = { 64, 128, 256, 512, 1024, 2048 };
	for (int r = 0; r < 6; r++)
	{
		int len = R[r];
		double* matrix1 = (double*)_aligned_malloc(sizeof(double)* len * len, 16);
		double* matrix2 = (double*)_aligned_malloc(sizeof(double)* len* len, 16);
		double* res = (double*)_aligned_malloc(sizeof(double)* len*len, 16);
		double* res2 = (double*)_aligned_malloc(sizeof(double)* len*len, 16);

		double* res_sse = (double*)_aligned_malloc(sizeof(double)* len*len, 16);
		double* res2_sse = (double*)_aligned_malloc(sizeof(double)* len*len, 16);

		for (int i = 0; i < len; ++i){
			for (int j = 0; j < len; ++j){
				matrix1[i * len + j] = (double)(rand() % 3);
				matrix2[i * len + j] = (double)(rand() % 3);
				res[i*len + j] = 0.0;
				res2[i*len + j] = 0.0;
				res_sse[i*len + j] = 0.0;
				res2_sse[i*len + j] = 0.0;
			}
		}

		_tcprintf(_T("-------------- %d ---------------\n"), len);
		StartCounter();
		matrix_mul_matrix(matrix1, matrix2, res, len);
		double time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d: %f ms\n"), len, len, time_end);

		StartCounter();
		omp_matrix_mul_matrix(matrix1, matrix2, res2, len);
		time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d: %f ms (OMP)\n"), len, len, time_end);

		_tcprintf(_T("--Count of different elements: %d\n"), compare_matrix(res, res2, len));
		
		StartCounter();
		matrix_mul_matrix_sse(matrix1, matrix2, res_sse, len);
		time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d: %f ms (SSE)\n"), len, len, time_end);

		StartCounter();
		omp_matrix_mul_matrix_sse(matrix1, matrix2, res2_sse, len);
		time_end = GetCounter();
		_tcprintf(_T("Time for multiple matrix %d x %d: %f ms (SSE OMP)\n"), len, len, time_end);
		_tcprintf(_T("--Count of different elements: %d\n"), compare_matrix(res_sse, res2_sse, len));

		
		if (avxSupported){
			double* res3 = (double*)_aligned_malloc(sizeof(double)* len*len, 16);
			double* res3_omp = (double*)_aligned_malloc(sizeof(double)* len*len, 16);
			for (int i = 0; i < len; ++i){
				for (int j = 0; j < len; ++j){
					res3[i*len + j] = 0.0;
					res3_omp[i*len + j] = 0.0;
				}
			}
			StartCounter();
			matrix_mul_matrix_avx(matrix1, matrix2, res3, len);
			time_end = GetCounter();
			_tcprintf(_T("Time for multiple matrix %d x %d: %f ms (AVX)\n"), len, len, time_end);

			StartCounter();
			omp_matrix_mul_matrix_avx(matrix1, matrix2, res3_omp, len);
			time_end = GetCounter();
			_tcprintf(_T("Time for multiple matrix %d x %d: %f ms (AVX OMP)\n"), len, len, time_end);
			_tcprintf(_T("--Count of different elements: %d\n"), compare_matrix(res3_omp, res3, len));
			_aligned_free(res3);
			_aligned_free(res3_omp);
		}
		_aligned_free(matrix1);
		_aligned_free(matrix2);
		_aligned_free(res);
		_aligned_free(res2);
		_aligned_free(res_sse);
		_aligned_free(res2_sse);
	}
}
#pragma endregion

int _tmain(int argc, _TCHAR* argv[])
{
	srand(time(0));
	avx_support();
	//test_max();
	//test_mul_vector();
	test_mul_matrix();
	return 0;
}

