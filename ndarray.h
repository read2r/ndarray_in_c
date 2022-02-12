#ifndef __ND_ARRAY_H__
#define __ND_ARRAY_H__
#include "ndshape.h"

typedef struct tagNdArray NdArray;

typedef enum tagDataType {
    DT_INT = 1342,
    DT_FLOAT,
    DT_DOUBLE,
} DataType;

typedef struct tagNdArray {
    DataType datatype;
    NdShape *shape;
    unsigned int item_size;
    unsigned int size;
    void *data;
} NdArray;

typedef void*(*broadcast_func)(void*);

// create, deep copy and delete ndarray
NdArray* NdArray_new(void *data, NdShape *ndshape, DataType datatype);
NdArray* NdArray_copy(NdArray *src);
void NdArray_free(NdArray **dptr_ndarray);
void NdArray_sub_free(NdArray **dptr_ndarray);

// just array
NdArray* NdArray_zeros(unsigned int len, DataType datatype);
NdArray* NdArray_ones(unsigned int len, DataType datatype);
NdArray* NdArray_arange(unsigned int start, unsigned int end, DataType datatype);

// reshape ndarray
int NdArray_reshape(NdArray *ndarray, NdShape *ndshape);

// get, set element, ndarray
void* NdArray_getAt(NdArray *ndarray, unsigned int *position);
void NdArray_setAt(NdArray *ndarray, unsigned int *position, void* data);
NdArray* NdArray_subarray(NdArray *ndarray, unsigned int *position, int pdim);

// print
void NdArray_printArray(NdArray *ndarray);
void NdArray_printShape(NdArray *ndarray);

// matrix operations
NdArray* NdArray_dot(NdArray *a, NdArray *b);
NdArray* NdArray_matmul(NdArray *a, NdArray *b);
int NdArray_transpose(NdArray *ndarray); // Not yet implemented

int NdArray_add(NdArray *dest, NdArray *src);
int NdArray_sub(NdArray *dest, NdArray *src);
int NdArray_mul(NdArray *dest, NdArray *src);
int NdArray_div(NdArray *dest, NdArray *src);
//int NdArray_mod(NdArray *dest, NdArray *src);

// broadcast scalar
void NdArray_add_scalar(NdArray *ndarray, double value);
void NdArray_sub_scalar(NdArray *ndarray, double value);
void NdArray_mul_scalar(NdArray *ndarray, double value);
void NdArray_div_scalar(NdArray *ndarray, double value);
//void NdArray_mod_scalar(NdArray *ndarray, int value);

void NdArray_broadcast(NdArray *ndarray, broadcast_func bfunc);

#endif
