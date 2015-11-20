#include "mat.h"

//	Reading
mxArray *matGetMatrixInFile(const char *fileName, const char *matrixName);
mxArray *hdf5GetArrayFromFile(const char *fileName, const char *matrixName);

//	Writing
int matPutMatrixInFile(const char *fileName, const char *matrixName, const mxArray *matrix);
void hdf5PutArrayInFile(const char *fileName, const char *matrixName, const mxArray *matrix);