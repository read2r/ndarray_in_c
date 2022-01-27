#ifndef __ND_ARRAY_H__
#define __ND_ARRAY_H__
#include "ndshape.h"

typedef struct tagNdArray NdArray;

// Not yet used...
typedef enum tagDataType {
    DT_INT,
    DT_FLOAT,
    DT_DOUBLE,
} DataType;

typedef struct tagNdArray {
    //DataType datatype;
    NdShape *shape;
    unsigned int size;
    void *data;
} NdArray;

// create, deep copy and delete ndarray
NdArray* NdArray_new(void *data, NdShape *ndshape);
NdArray* NdArray_copy(NdArray *src);
void NdArray_free(NdArray **dptr_ndarray);

// just array
NdArray* NdArray_zeros(unsigned int len);
NdArray* NdArray_ones(unsigned int len);
NdArray* NdArray_arange(unsigned int start, unsigned int end);

// reshape ndarray
int NdArray_reshape(NdArray *ndarray, NdShape *ndshape);

// get element in ndarray
int NdArray_getAt(NdArray *ndarray, unsigned int *position);
void NdArray_setAt(NdArray *ndarray, unsigned int *position, void* data);

// print
void NdArray_printArray(NdArray *ndarray);
void NdArray_printShape(NdArray *ndarray);

// array & matrix operations
int NdArray_add(NdArray *dest, NdArray *src);
int NdArray_sub(NdArray *dest, NdArray *src);
int NdArray_mul(NdArray *dest, NdArray *src);
int NdArray_div(NdArray *dest, NdArray *src);
int NdArray_mod(NdArray *dest, NdArray *src);
NdArray* NdArray_dot(NdArray *a, NdArray *b);
NdArray* NdArray_matmul(NdArray *a, NdArray *b);
int NdArray_transpose(NdArray *ndarray); // Not yet implemented

// scalar operations
void NdArray_add_scalar(NdArray *ndarray, int value);
void NdArray_sub_scalar(NdArray *ndarray, int value);
void NdArray_mul_scalar(NdArray *ndarray, int value);
void NdArray_div_scalar(NdArray *ndarray, int value);
void NdArray_mod_scalar(NdArray *ndarray, int value);

#endif
