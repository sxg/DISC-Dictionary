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


//	Helpers
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
double artConc(const double artFrac);

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

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
	mxArray *A = matGetMatrixInFile("A.mat", "A");
	mxArray *B = matGetMatrixInFile("B.mat", "B");
	mxArray *aCol1 = matGetColInMatrix(A, 0);

	double *aCol1Data = mxGetPr(aCol1);
	for (int i = 0; i < mxGetM(aCol1); i++) {
		printf("%f ", aCol1Data[i]);
	}
	
	artConc(aCol1);

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
	getchar();

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
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
