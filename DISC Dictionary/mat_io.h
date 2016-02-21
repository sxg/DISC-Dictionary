#include "mat.h"

//	Reading
mxArray *matGetMatrixInFile(const char *fileName, const char *matrixName);
mxArray *matGetColInMatrix(const mxArray *matrix, const int colIdx);
mxArray *matGetRowInMatrix(const mxArray *matrix, const int rowIdx);
mxArray *hdf5GetArrayFromFile(const char *fileName, const char *matrixName);

//	Writing
int matPutMatrixInFile(const char *fileName, const char *matrixName, const mxArray *matrix);
void hdf5PutArrayInFile(const char *fileName, const char *matrixName, const mxArray *matrix);

double *rowMajorMatrixFromColMajorMatrix(const double *colMajorMatrix, const mwIndex width, const mwIndex height);