#ifndef __ND_ARRAY_H__
#define __ND_ARRAY_H__
#include "ndshape.h"

typedef struct _NdArray NdArray;

typedef enum _DataType {
    DT_INT,
    DT_DOUBLE,
    DT_BOOL,
} DataType;

typedef enum _CompareTag {
    CT_GT,
    CT_GE,
    CT_LT,
    CT_LE,
    CT_EQ,
} CompareTag;

typedef struct _NdArray {
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
NdArray* NdArray_random(unsigned int len, DataType datatype);
NdArray* NdArray_random_range(unsigned int len, unsigned int low, unsigned int high, DataType datatype);
NdArray* NdArray_random_gaussian(unsigned int len);
NdArray* NdArray_choice(unsigned int pick_len, unsigned int len, DataType datatype);

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
NdArray* NdArray_transpose(NdArray *ndarray);

NdArray* NdArray_suffle(NdArray *array);

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

// g : graeter
// ge : grater or equal
// l : less
// le : less or equal
// e : equal
NdArray* NdArray_compare(NdArray *a, NdArray *b, CompareTag ct);
NdArray* NdArray_compare_scalar(NdArray *self, double value, CompareTag ct);
NdArray* NdArray_mask(NdArray *self, NdArray* mask);

int NdArray_sum_int(NdArray *ndarray);
double NdArray_sum_double(NdArray *ndarray);
void* NdArray_sum(NdArray *ndarray);

NdArray* NdArray_sum_axis(NdArray *ndarray, unsigned int axis);
NdArray* NdArray_argmax_axis(NdArray *self, unsigned int axis);

int NdArray_max_int(NdArray *ndarray);
double NdArray_max_double(NdArray *ndarray);
void* NdArray_max(NdArray *ndarray);

int NdArray_min_int(NdArray *ndarray);
double NdArray_min_double(NdArray *ndarray);
void* NdArray_min(NdArray *ndarray);

double NdArray_mean_int(NdArray *ndarray);
double NdArray_mean_double(NdArray *ndarray);
void* NdArray_mean(NdArray *ndarray);

int NdArray_argmax(NdArray *ndarray);
int NdArray_argmin(NdArray *ndarray);

void NdArray_broadcast(NdArray *ndarray, broadcast_func bfunc);
void NdArray_convert_type(NdArray **ptr_self, DataType datatype);

#endif
