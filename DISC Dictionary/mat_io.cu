#include "mat_io.h"
#include <stdlib.h>
#include <string.h>
#include "hdf5.h"

//	Helpers

mwIndex rowMajorIndexFromColMajorIndex(const mwIndex colMajorIndex, const mwIndex width, const mwIndex height);
mwIndex colMajorIndexFromRowMajorIndex(const mwIndex rowMajorIndex, const mwIndex width, const mwIndex height);

double *rowMajorMatrixFromColMajorMatrix(const double *colMajorMatrix, const mwIndex width, const mwIndex height);
double *colMajorMatrixFromRowMajorMatrix(const double *rowMajorMatrix, const mwIndex width, const mwIndex height);


//	Reading

mxArray *matGetMatrixInFile(const char *fileName, const char *matrixName) {
	MATFile *matrixFile = matOpen(fileName, "r");
	mxArray *matrix = matGetVariable(matrixFile, matrixName);
	matClose(matrixFile);
	return matrix;
}

mxArray *matGetColInMatrix(const mxArray *matrix, const int colIdx) {
	double *matrixData = mxGetPr(matrix);
	int numRows = mxGetM(matrix);
	int numCols = mxGetN(matrix);
	double *matrixColData = (double *)mxMalloc(sizeof(double) * numRows);
	for (int i = colIdx * numRows; i < colIdx * numRows + numRows; i++) {
		matrixColData[i % numRows] = matrixData[i];
	}
	
	mxArray *matrixCol = mxCreateDoubleMatrix(numRows, 1, mxREAL);
	mxSetPr(matrixCol, matrixColData);
	return matrixCol;
}

mxArray *matGetRowInMatrix(const mxArray *matrix, const int rowIdx) {
	double *matrixData = mxGetPr(matrix);
	int numRows = mxGetM(matrix);
	int numCols = mxGetN(matrix);
	double *matrixRowData = (double *)mxMalloc(sizeof(double) * numCols);
	for (int i = rowIdx; i < numRows * numCols; i += numRows) {
		matrixRowData[i / numRows] = matrixData[i];
	}

	mxArray *matrixRow = mxCreateDoubleMatrix(1, numCols, mxREAL);
	mxSetPr(matrixRow, matrixRowData);
	return matrixRow;
}

mxArray *hdf5GetArrayFromFile(const char *fileName, const char *matrixName) {
	hid_t file_id = H5Fopen(fileName, H5F_ACC_RDWR, H5P_DEFAULT);
	hid_t dataset_id = H5Dopen2(file_id, matrixName, H5P_DEFAULT);

	hid_t dataspace_id = H5Dget_space(dataset_id);
	hsize_t dims[2];
	H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
	
	double *dset_data = (double *)malloc(sizeof(double) * dims[0] * dims[1]);
	H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);
	H5Dclose(dataset_id);

	mxArray *matrix = mxCreateDoubleMatrix(dims[1], dims[0], mxREAL);
	mxSetData(matrix, dset_data);
	return matrix;
}


//	Writing

int matPutMatrixInFile(const char *fileName, const char *matrixName, const mxArray *matrix) {
	MATFile *matrixFile = matOpen(fileName, "w");
	int err = matPutVariable(matrixFile, matrixName, matrix);
	matClose(matrixFile);
	return err;
}

void hdf5PutArrayInFile(const char *fileName, const char *matrixName, const mxArray *matrix) {
	char *datasetName = (char *)malloc(strlen(matrixName) + 2);
	datasetName[0] = '/';
	datasetName[1] = '\0';
	strcat(datasetName, matrixName);

	hid_t file_id = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dims[] = { mxGetM(matrix), mxGetN(matrix) };
	hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
	hid_t dataset_id = H5Dcreate2(file_id, datasetName, H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mxGetPr(matrix));

	H5Dclose(dataset_id);
	H5Sclose(dataspace_id);
	H5Fclose(file_id);
}



//	Helpers

mwIndex rowMajorIndexFromColMajorIndex(const mwIndex colMajorIndex, const mwIndex width, const mwIndex height) {
	mwIndex row = colMajorIndex % height;
	mwIndex col = colMajorIndex / height;
	return row * width + col;
}

mwIndex colMajorIndexFromRowMajorIndex(const mwIndex rowMajorIndex, const mwIndex width, const mwIndex height) {
	return rowMajorIndexFromColMajorIndex(rowMajorIndex, height, width);
}

double *rowMajorMatrixFromColMajorMatrix(const double *colMajorMatrix, const mwIndex width, const mwIndex height) {
	int colMajorIndex, rowMajorIndex;
	double *rowMajorMatrix = (double *)malloc(sizeof(double) * width * height);

	for (colMajorIndex = 0; colMajorIndex < width * height; colMajorIndex++) {
		rowMajorIndex = rowMajorIndexFromColMajorIndex(colMajorIndex, width, height);
		rowMajorMatrix[rowMajorIndex] = colMajorMatrix[colMajorIndex];
	}
	return rowMajorMatrix;
}

double *colMajorMatrixFromRowMajorMatrix(const double *rowMajorMatrix, const mwIndex width, const mwIndex height) {
	int colMajorIndex, rowMajorIndex;
	double *colMajorMatrix = (double *)malloc(sizeof(double) * width * height);

	for (rowMajorIndex = 0; rowMajorIndex < width * height; rowMajorIndex++) {
		colMajorIndex = colMajorIndexFromRowMajorIndex(rowMajorIndex, width, height);
		colMajorMatrix[colMajorIndex] = rowMajorMatrix[rowMajorIndex];
	}
	return colMajorMatrix;
}