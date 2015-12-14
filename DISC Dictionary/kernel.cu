#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include "mat_io.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <string.h>

//	Constants
const double ALPHA = 15.0f * M_PI / 180.0f;
const int BASE_FRAME = 3;
const double TR = 4.54f;
const double T10b = 1.8f * 1000.0f;
const double T10p = 1.8f * 1000.0f;
const double T10L = 0.747f * 1000.0f;
const double R10b = 1.0f / T10b;
const double R10p = 1.0f / T10p;
const double R10L = 1.0f / T10L;
const double HCT = 0.4f;
const double RELAXIVITY = 4.5f;


double *artConc(const mxArray *artFrac) {
	//	Calculate S0b
	int numRows = mxGetM(artFrac);
	const double *artFracData = mxGetPr(artFrac);
	double S0b = artFracData[BASE_FRAME] * ((1.0f - exp(-1.0f * R10b * TR) * cos(ALPHA)) / (1.0f - exp(-1.0f * R10b * TR)) / sin(ALPHA));
	
	//	Calculate R1b
	double *R1b = new double[numRows];
	for (int i = 0; i < numRows; i++) {
		R1b[i] = log10( ( (S0b * sin(ALPHA)) - (artFracData[i] * cos(ALPHA)) ) / (S0b * sin(ALPHA) - artFracData[i]) ) / TR;
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
		R1p[i] = log10( ( (S0p * sin(ALPHA)) - (pvData[i] * cos(ALPHA)) ) / (S0p * sin(ALPHA) - pvData[i]) ) / TR;
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
		R1L[i] = log10(((S0L * sin(ALPHA)) - (liverData[i] * cos(ALPHA))) / (S0L * sin(ALPHA) - liverData[i])) / TR;
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
	double *C = new double[n];

	for (int i = 0; i < n; i++) {
		double sum = 0.0f;
		for (int j = 0; j < i; j++) {
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

int fiveDimIdxToLinIdx(int i, int size_i, int j, int size_j, int k, int size_k, int l, int size_l, int m) {
	return i + (j * size_i) + (k * size_i * size_j) + (l * size_i * size_j *size_k) + (m * size_i * size_j * size_k * size_l);
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
	fiveDimIdx[4] = idx / size_l;

	return fiveDimIdx;
}

__global__ void popDict(double **dict, const double *times, const double *artConc, const double *pvConc, const int n, const double *AF, const double *MTT, const double *DV, const double *t1, const double *t2) {
	int *fiveDimIdx = linIdxToFiveDimIdx(threadIdx.x, 21, 21, 21, 26);
	double *disc_out = new double[n];
	disc_out = disc(times, artConc, pvConc, n, AF[fiveDimIdx[0]], MTT[fiveDimIdx[1]], DV[fiveDimIdx[2]], t1[fiveDimIdx[3]], t2[fiveDimIdx[4]]);
	dict[threadIdx.x] = disc_out;
}

int main()
{
	mxArray *allts = matGetMatrixInFile("data.mat", "allts");
	mxArray *times = matGetColInMatrix(allts, 0);
	mxArray *AF = matGetColInMatrix(allts, 1);
	mxArray *PV = matGetColInMatrix(allts, 2);
	mxArray *Liver = matGetColInMatrix(allts, 3);

	int n = mxGetM(times);

	double *timesData = mxGetPr(times);
	double *Cb_plasma = artConc(AF);
	double *Cp_plasma = pvConc(PV);
	
	double *AF_range = linspace(0.0f, 100.0f, 21);
	double *MTT_range = linspace(0.0f, 100.0f, 21);
	double *DV_range = linspace(0.0f, 100.0f, 21);
	double *t1_range = linspace(0.0f, 5.0f, 26);
	double *t2_range = linspace(0.0f, 5.0f, 26);

	double **dict = (double **)malloc(sizeof(double *) * 21 * 21 * 21 * 26 * 26);

	double *d_timesData;
	cudaMalloc(&d_timesData, sizeof(double) * n);
	cudaMemcpy(d_timesData, timesData, sizeof(double) * n, cudaMemcpyHostToDevice);

	double *d_Cb_plasma, *d_Cp_plasma;
	cudaMalloc(&d_Cb_plasma, sizeof(double) * n);
	cudaMalloc(&d_Cp_plasma, sizeof(double) * n);
	cudaMemcpy(d_Cb_plasma, Cb_plasma, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Cp_plasma, Cp_plasma, sizeof(double) * n, cudaMemcpyHostToDevice);

	double *d_AF_range, *d_MTT_range, *d_DV_range, *d_t1_range, *d_t2_range;
	cudaMalloc(&d_AF_range, sizeof(double) * 21);
	cudaMalloc(&d_MTT_range, sizeof(double) * 21);
	cudaMalloc(&d_DV_range, sizeof(double) * 21);
	cudaMalloc(&d_t1_range, sizeof(double) * 26);
	cudaMalloc(&d_t2_range, sizeof(double) * 26);
	cudaMemcpy(d_AF_range, AF_range, sizeof(double) * 21, cudaMemcpyHostToDevice);
	cudaMemcpy(d_MTT_range, MTT_range, sizeof(double) * 21, cudaMemcpyHostToDevice);
	cudaMemcpy(d_DV_range, DV_range, sizeof(double) * 21, cudaMemcpyHostToDevice);
	cudaMemcpy(d_t1_range, t1_range, sizeof(double) * 26, cudaMemcpyHostToDevice);
	cudaMemcpy(d_t2_range, t2_range, sizeof(double) * 26, cudaMemcpyHostToDevice);
	
	double **d_dict;
	cudaMalloc(&d_dict, sizeof(double *) * 21 * 21 * 21 * 26 * 26);

	popDict<<< 1, 21 * 21 * 21 * 26 * 26 >>>(d_dict, d_timesData, d_Cb_plasma, d_Cp_plasma, n, d_AF_range, d_MTT_range, d_DV_range, d_t1_range, d_t2_range);

	cudaMemcpy(dict, d_dict, sizeof(double *) * 21 * 21 * 21 * 26 * 26, cudaMemcpyDeviceToHost);
	double *linearDict = (double *)malloc(sizeof(double) * 21 * 21 * 21 * 26 * 26 * n);
	for (int i = 0; i < 21 * 21 * 21 * 26 * 26; i++) {
		double *timeSeries = dict[i];
		for (int j = 0; j < n; j++) {
			double temp = timeSeries[j];
			linearDict[i * n + j] = temp;
		}
	}

	mxArray *linearDictMatrix = mxCreateDoubleMatrix(21 * 21 * 21 * 26 * 26, n, mxREAL);
	matPutMatrixInFile("Dictionary.mat", "Dictionary", linearDictMatrix);

	//const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

	//	Pause
	//	getchar();

	cudaFree(d_dict);
	cudaFree(d_timesData);
	cudaFree(d_Cb_plasma);
	cudaFree(d_Cp_plasma);
	cudaFree(d_AF_range);
	cudaFree(d_MTT_range);
	cudaFree(d_DV_range);
	cudaFree(d_t1_range);
	cudaFree(d_t2_range);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
