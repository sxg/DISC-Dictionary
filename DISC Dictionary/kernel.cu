#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include "mat_io.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <string.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
		        } \
	    } while (0)

//	Constants
const double ALPHA = 15.0f * M_PI / 180.0f;
const int BASE_FRAME = 2;
const double TR = 4.54f;
const double T10b = 1.664f * 1000.0f;
const double T10p = 1.584f * 1000.0f;
const double T10L = 0.8f * 1000.0f;
const double R10b = 1.0f / T10b;
const double R10p = 1.0f / T10p;
const double R10L = 1.0f / T10L;
const double HCT = 0.4f;
const double RELAXIVITY = 6.3f;


double *artConc(const mxArray *artFrac) {
	//	Calculate S0b
	int numRows = mxGetM(artFrac);
	const double *artFracData = mxGetPr(artFrac);
	double S0b = artFracData[BASE_FRAME] * ((1.0f - exp(-1.0f * R10b * TR) * cos(ALPHA)) / (1.0f - exp(-1.0f * R10b * TR)) / sin(ALPHA));

	//	Calculate R1b
	double *R1b = new double[numRows];
	for (int i = 0; i < numRows; i++) {
		R1b[i] = log( ( (S0b * sin(ALPHA)) - (artFracData[i] * cos(ALPHA)) ) / (S0b * sin(ALPHA) - artFracData[i]) ) / TR;
	}
	
	//	Calculate Cb_artery
	double *Cb_artery = new double[numRows];
	for (int i = 0; i < numRows; i++) {
		Cb_artery[i] = (R1b[i] - R10b) * 1000.0f / RELAXIVITY;
	}

	//	Calculate Cb_plasma
	double *Cb_plasma = new double[numRows];
	for (int i = 0; i < numRows; i++) {
		Cb_plasma[i] = Cb_artery[i] / (1.0f - HCT);
	}

	return Cb_plasma;
}

double *pvConc(const mxArray *pv) {
	//	Calculate S0p
	int numRows = mxGetM(pv);
	const double *pvData = mxGetPr(pv);
	double S0p = pvData[BASE_FRAME] * ((1.0f - exp(-1.0f * R10p * TR) * cos(ALPHA)) / (1.0f - exp(-1.0f * R10p * TR)) / sin(ALPHA));

	//	Calculate R1p
	double *R1p = new double[numRows];
	for (int i = 0; i < numRows; i++) {
		R1p[i] = log( ( (S0p * sin(ALPHA)) - (pvData[i] * cos(ALPHA)) ) / (S0p * sin(ALPHA) - pvData[i]) ) / TR;
	}

	//	Calculate Cp_artery
	double *Cp_artery = new double[numRows];
	for (int i = 0; i < numRows; i++) {
		Cp_artery[i] = (R1p[i] - R10p) * 1000.0f / RELAXIVITY;
	}

	//	Calculate Cp_plasma
	double *Cp_plasma = new double[numRows];
	for (int i = 0; i < numRows; i++) {
		Cp_plasma[i] = Cp_artery[i] / (1.0f - HCT);
	}

	return Cp_plasma;
}

double *clearance(const mxArray *liver) {
	//	Calculate S0L
	int numRows = mxGetM(liver);
	const double *liverData = mxGetPr(liver);
	double S0L = liverData[BASE_FRAME] * ((1.0f - exp(-1.0f * R10L * TR) * cos(ALPHA)) / (1.0f - exp(-1.0f * R10L * TR)) / sin(ALPHA));
	
	//	Calculate R1L
	double *R1L = new double[numRows];
	for (int i = 0; i < numRows; i++) {
		R1L[i] = log(((S0L * sin(ALPHA)) - (liverData[i] * cos(ALPHA))) / (S0L * sin(ALPHA) - liverData[i])) / TR;
	}

	//	Calculate CL
	double *CL = new double[numRows];
	for (int i = 0; i < numRows; i++) {
		CL[i] = (R1L[i] - R10L) * 1000.0f / RELAXIVITY * 0.2627f;
	}

	return CL;
}

__device__ double *disc(const double *times, const double *artConc, const double *pvConc, const int n, const double AF, const double DV, const double MTT, const double t1, const double t2) {
	double k1a = AF * DV / MTT;
	double k1p = DV * (1.0f - AF) / MTT;
	double k2 = 1.0f / MTT;
	double dt = times[1] - times[0];
	double *C = (double *)malloc(sizeof(double) * n);

	for (int i = 0; i < n; i++) {
		double sum = 0.0f;
		for (int j = 0; j <= i; j++) {
			double sum1 = 0.0f;
			if (round(j - t1 * 1000.0f) > 0.0f) {
				sum1 += k1a * artConc[(int)round(j - t1 * 1000.0f)];
			}

			if (round(j - t2) > 0.0f) {
				sum1 += k1p * pvConc[(int)round(j - t2)];
			}

			sum += sum1 * exp(-1.0f * k2 * (i - j) * dt) * dt;
		}
		C[i] = sum;
	}

	return C;
}

double *linspace(double start, double end, int n) {
	double *array = new double[n];
	double step = (end - start) / (n - 1);
	for (int i = 0; i < n; i++) {
		array[i] = start + (i * step);
	}
	return array;
}

__device__ int multiDimIdxToLinIdx(int *idxs, int *sizes, int nDims) {
	int linIdx = 0;
	for (int i = 0; i < nDims; i++) {
		int sizeProduct = 1;
		for (int j = 0; j < i; j++) {
			sizeProduct *= sizes[j];
		}
		linIdx += idxs[i] * sizeProduct;
	}
	return linIdx;
}

__device__ int fiveDimIdxToLinIdx(int i, int size_i, int j, int size_j, int k, int size_k, int l, int size_l, int m) {
	return i + (j * size_i) + (k * size_i * size_j) + (l * size_i * size_j *size_k) + (m * size_i * size_j * size_k * size_l);
}

int fiveDimIdxToLinIdxDev(int i, int size_i, int j, int size_j, int k, int size_k, int l, int size_l, int m) {
	return i + (j * size_i) + (k * size_i * size_j) + (l * size_i * size_j *size_k) + (m * size_i * size_j * size_k * size_l);
}

__device__ int *linIdxToMultiDimIdx(int idx, int *sizes, int nDims) {
	int *multiDimIdx = new int[nDims];

	for (int i = 0; i < nDims; i++) {
		if (i == 0) {
			multiDimIdx[i] = idx % sizes[i];
		}
		else if (i == nDims - 1) {
			multiDimIdx[i] = idx / sizes[i - 1];
		}
		else {
			multiDimIdx[i] = (idx / sizes[i - 1]) % sizes[i];
		}

		idx -= multiDimIdx[i];
	}

	return multiDimIdx;
}

__device__ int *linIdxToFiveDimIdx(int idx, int size_i, int size_j, int size_k, int size_l) {
	int *fiveDimIdx = new int[5];

	//	Get first dim
	fiveDimIdx[0] = idx % size_i;

	//	Get second dim
	idx -= fiveDimIdx[0];
	fiveDimIdx[1] = (idx / size_i) % size_j;

	//	Get third dim
	idx -= fiveDimIdx[1];
	fiveDimIdx[2] = (idx / size_j) % size_k;

	//	Get fourth dim
	idx -= fiveDimIdx[2];
	fiveDimIdx[3] = (idx / size_k) % size_l;

	//	Get fifth dim
	idx -= fiveDimIdx[3];
	fiveDimIdx[4] = (idx / size_l);

	return fiveDimIdx;
}

__global__ void popDict(double *dict, const double *times, const double *artConc, const double *pvConc, const int n, const double *AF, const double *DV, const double *MTT, const double *t1, const double *t2) {
	//int *fiveDimIdx = linIdxToFiveDimIdx(blockIdx.x, 21, 21, 21, 26);

	const double AFx = AF[blockIdx.x];
	const double DVx = DV[blockIdx.y];
	const double MTTx = MTT[blockIdx.z];
	const double t1x = t1[threadIdx.x];
	const double t2x = t2[threadIdx.y];
	int linIdx = fiveDimIdxToLinIdx(blockIdx.x, 21, blockIdx.y, 21, blockIdx.z, 21, threadIdx.x, 26, threadIdx.y);
	
	double k1a = AFx * DVx / MTTx;
	double k1p = DVx * (1.0f - AFx) / MTTx;
	double k2 = 1.0f / MTTx;
	double dt = times[1] - times[0];

	for (int i = 0; i < n; i++) {
		double sum = 0.0f;
		for (int j = 0; j <= i; j++) {
			double sum1 = 0.0f;
			if (round(j - t1x * 1000.0f) > 0.0f) {
				sum1 += k1a * artConc[(int)round(j - t1x * 1000.0f)];
			}

			if (round(j - t2x) > 0.0f) {
				sum1 += k1p * pvConc[(int)round(j - t2x)];
			}

			sum += sum1 * exp(-1.0f * k2 * (i - j) * dt) * dt;
		}
		dict[linIdx * n + i] = sum;
	}
	
	/*double k1a = AF[blockIdx.x] * DV[blockIdx.y] / MTT[blockIdx.z];
	double k1p = DV[blockIdx.y] * (1.0f - AF[blockIdx.x]) / MTT[blockIdx.z];
	double k2 = 1.0f / MTT[blockIdx.z];
	double dt = times[1] - times[0];

	for (int i = 0; i < n; i++) {
		double sum = 0.0f;
		for (int j = 0; j <= i; j++) {
			double sum1 = 0.0f;
			if (round(j - t1[threadIdx.x] * 1000.0f) > 0.0f) {
				sum1 += k1a * artConc[(int)round(j - t1[threadIdx.x] * 1000.0f)];
			}

			if (round(j - t2[threadIdx.y]) > 0.0f) {
				sum1 += k1p * pvConc[(int)round(j - t2[threadIdx.y])];
			}

			sum += sum1 * exp(-1.0f * k2 * (i - j) * dt) * dt;
		}
		dict[linIdx * n + i] = sum;
	}*/
	
	/*double *disc_out = disc(times, artConc, pvConc, n, AF[blockIdx.x], MTT[blockIdx.y], DV[blockIdx.z], t1[threadIdx.x], t2[threadIdx.y]);
	for (int i = 0; i < n; i++) {
		dict[linIdx * n + i] = disc_out[i];
	}*/
}

int main()
{
	mxArray *allts = matGetMatrixInFile("data2.mat", "allts");
	mxArray *times = matGetColInMatrix(allts, 0);
	mxArray *AF = matGetColInMatrix(allts, 1);
	mxArray *PV = matGetColInMatrix(allts, 2);
	mxArray *Liver = matGetColInMatrix(allts, 3);

	int n = mxGetM(times);

	double *timesData = mxGetPr(times);
	double *Cb_plasma = artConc(AF);
	double *Cp_plasma = pvConc(PV);
	
	double *AF_range = linspace(0.01f, 1.0f, 21);
	double *DV_range = linspace(0.01f, 1.0f, 21);
	double *MTT_range = linspace(1.0f, 100.0f, 21);
	double *t1_range = linspace(0.001f, 0.02f, 26);
	double *t2_range = linspace(0.001f, 0.02f, 26);

	printf("Mallocing...\n");
	double *dict = (double *)mxMalloc(sizeof(double) * 21 * 21 * 21 * 26 * 26 * n);
	printf("Done mallocing.\n");

	double *d_timesData;
	cudaMalloc(&d_timesData, sizeof(double) * n);
	cudaMemcpy(d_timesData, timesData, sizeof(double) * n, cudaMemcpyHostToDevice);

	double *d_Cb_plasma, *d_Cp_plasma;
	cudaMalloc(&d_Cb_plasma, sizeof(double) * n);
	cudaMalloc(&d_Cp_plasma, sizeof(double) * n);
	cudaMemcpy(d_Cb_plasma, Cb_plasma, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Cp_plasma, Cp_plasma, sizeof(double) * n, cudaMemcpyHostToDevice);

	double *d_AF_range, *d_DV_range, *d_MTT_range, *d_t1_range, *d_t2_range;
	cudaMalloc(&d_AF_range, sizeof(double) * 21);
	cudaMalloc(&d_DV_range, sizeof(double) * 21);
	cudaMalloc(&d_MTT_range, sizeof(double) * 21);
	cudaMalloc(&d_t1_range, sizeof(double) * 26);
	cudaMalloc(&d_t2_range, sizeof(double) * 26);
	cudaMemcpy(d_AF_range, AF_range, sizeof(double) * 21, cudaMemcpyHostToDevice);
	cudaMemcpy(d_DV_range, DV_range, sizeof(double) * 21, cudaMemcpyHostToDevice);
	cudaMemcpy(d_MTT_range, MTT_range, sizeof(double) * 21, cudaMemcpyHostToDevice);
	cudaMemcpy(d_t1_range, t1_range, sizeof(double) * 26, cudaMemcpyHostToDevice);
	cudaMemcpy(d_t2_range, t2_range, sizeof(double) * 26, cudaMemcpyHostToDevice);
	
	double *d_dict;
	cudaMalloc(&d_dict, sizeof(double) * 21 * 21 * 21 * 26 * 26 * n);
	cudaCheckErrors("cudaMalloc d_dict");
	
	printf("Launching kernel...\n");
	popDict <<< dim3(21, 21, 21), dim3(26, 26) >>>(d_dict, d_timesData, d_Cb_plasma, d_Cp_plasma, n, d_AF_range, d_DV_range, d_MTT_range, d_t1_range, d_t2_range);
	cudaCheckErrors("launch kernel fail");
	cudaDeviceSynchronize();
	cudaCheckErrors("cuda sync fail");
	printf("Kernel finished.\n");

	cudaMemcpy(dict, d_dict, sizeof(double) * 21 * 21 * 21 * 26 * 26 * n, cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy device to host");
	cudaFree(d_dict);

	mxArray *dictMatrix = mxCreateDoubleMatrix(21 * 21 * 21 * 26 * 26, n, mxREAL);
	mxSetPr(dictMatrix, dict);
	printf("Saving dictionary...\n");
	hdf5PutArrayInFile("HDF5_Dictionary.mat", "Dictionary", dictMatrix);
	printf("Done saving dictionary.\n");

	//	Pause
	getchar();

	cudaFree(d_timesData);
	cudaFree(d_Cb_plasma);
	cudaFree(d_Cp_plasma);
	cudaFree(d_AF_range);
	cudaFree(d_DV_range);
	cudaFree(d_MTT_range);
	cudaFree(d_t1_range);
	cudaFree(d_t2_range);

    return 0;
}
