#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "ndarray.h"
#include "ndshape.h"

NdArray* NdArray_new(void *data, NdShape *ndshape) {
    NdArray *ndarray = (NdArray*)malloc(sizeof(NdArray));
    ndarray->shape = NdShape_copy(ndshape);
    ndarray->size = sizeof(unsigned int) * ndarray->shape->len;
    ndarray->data = malloc(ndarray->size);
    if(data == NULL) {
        memset(ndarray->data, 0, ndarray->size);
    } else {
        memcpy(ndarray->data, data, ndarray->size);
    }
    return ndarray;
}

NdArray* NdArray_copy(NdArray *src) {
    NdArray *dest = NdArray_new(NULL, src->shape);
    dest->size = src->size;
    memcpy(dest->data, src->data, src->size);
    return dest;
}

void NdArray_free(NdArray **dptr_ndarray) {
    NdShape_free(&((*dptr_ndarray)->shape));
    free(*dptr_ndarray);
    *dptr_ndarray= NULL;
}

NdArray* NdArray_empty() {
    NdShape *ndshape = NdShape_empty(1);
    NdArray *ndarray = (NdArray*)malloc(sizeof(NdArray));
    return ndarray;
}

NdArray* NdArray_zeros(unsigned int len) {
    NdArray *ndarray = (NdArray*)malloc(sizeof(NdArray));
    ndarray->shape = NdShape_new(1, len);
    ndarray->size = sizeof(unsigned int) * ndarray->shape->len;
    ndarray->data = malloc(ndarray->size);

    memset(ndarray->data, 0, ndarray->size);
    return ndarray;
}

NdArray* NdArray_ones(unsigned int len) {
    NdArray *ndarray = NdArray_zeros(len);
    for(int i = 0; i < len; i++) {
        *((int*)ndarray->data + i) = 1;
    }
    return ndarray;
}

NdArray* NdArray_arange(unsigned int start, unsigned int end) {
    NdArray *ndarray = NdArray_zeros(end - start);
    for(int i = 0; i < end - start; i++) {
        *((int*)ndarray->data + i) = start + i;
    }
    return ndarray;
}

int NdArray_reshape(NdArray *ndarray, NdShape *ndshape) {
    return NdShape_reshape(ndarray->shape, ndshape);
}

int NdArray_getAt(NdArray *ndarray, unsigned int *position) {
    unsigned int offset = 0;
    unsigned int len = ndarray->shape->len;

    for(int i = 0; i < ndarray->shape->dim; i++) {
        len /= ndarray->shape->arr[i];
        offset += position[i] * len;
    }

    return *((int*)ndarray->data + offset);
}

void NdArray_setAt(NdArray *ndarray, unsigned int *position, void* data) {
    unsigned int offset = 0;
    unsigned int len = ndarray->shape->len;

    for(int i = 0; i < ndarray->shape->dim; i++) {
        len /= ndarray->shape->arr[i];
        offset += position[i] * len;
    }

    *((int*)ndarray->data + offset) = *(int*)data;
}

void printArray(NdArray *ndarray, unsigned int *position, int dim) {
    if(dim == ndarray->shape->dim) {
        printf("%d ", NdArray_getAt(ndarray, position));
        return;
    }

    printf("[ ");
    for(int i = 0; i < ndarray->shape->arr[dim]; i++) {
        position[dim] = i;
        printArray(ndarray, position, dim+1);
    }
    printf("] ");
}

void NdArray_printArray(NdArray *ndarray) {
    unsigned int *position = (unsigned int*)malloc(sizeof(unsigned int) * ndarray->shape->dim);
    memset(position, 0, sizeof(unsigned int) * ndarray->shape->dim);
    printArray(ndarray, position, 0);
    printf("\n");
}

void NdArray_printShape(NdArray *ndarray) {
    return NdShape_print(ndarray->shape);
}

int NdArray_add(NdArray *dest, NdArray *src) {
    if(!NdShape_compare(dest->shape, src->shape)) {
        return 0;
    }

    for(int i = 0; i < dest->shape->len; i++) {
        *((int*)dest->data + i) += *((int*)src->data + i);
    }

    return 1;
}

int NdArray_sub(NdArray *dest, NdArray *src) {
    if(!NdShape_compare(dest->shape, src->shape)) {
        return 0;
    }

    for(int i = 0; i < dest->shape->len; i++) {
        *((int*)dest->data + i) -= *((int*)src->data + i);
    }

    return 1;
}

int NdArray_mul(NdArray *dest, NdArray *src) {
    if(!NdShape_compare(dest->shape, src->shape)) {
        return 0;
    }

    for(int i = 0; i < dest->shape->len; i++) {
        *((int*)dest->data + i) *= *((int*)src->data + i);
    }

    return 1;
}

int NdArray_div(NdArray *dest, NdArray *src) {
    if(!NdShape_compare(dest->shape, src->shape)) {
        return 0;
    }

    for(int i = 0; i < dest->shape->len; i++) {
        *((int*)dest->data + i) /= *((int*)src->data + i);
    }

    return 1;
}

int NdArray_mod(NdArray *dest, NdArray *src) {
    if(!NdShape_compare(dest->shape, src->shape)) {
        return 0;
    }

    for(int i = 0; i < dest->shape->len; i++) {
        *((int*)dest->data + i) %= *((int*)src->data + i);
    }

    return 1;
}

void NdArray_scalarAdd(NdArray *ndarray, int value) {
    int *data = (int*)ndarray->data;
    for(int i = 0; i < ndarray->shape->len; i++) {
        data[i] += value;
    }
}

void NdArray_scalarSub(NdArray *ndarray, int value) {
    int *data = (int*)ndarray->data;
    for(int i = 0; i < ndarray->shape->len; i++) {
        data[i] -= value;
    }
}

void NdArray_scalarMul(NdArray *ndarray, int value) {
    int *data = (int*)ndarray->data;
    for(int i = 0; i < ndarray->shape->len; i++) {
        data[i] *= value;
    }
}

void NdArray_scalarDiv(NdArray *ndarray, int value) {
    assert(value != 0);
    int *data = (int*)ndarray->data;
    for(int i = 0; i < ndarray->shape->len; i++) {
        data[i] /= value;
    }
}

void NdArray_scalarMod(NdArray *ndarray, int value) {
    assert(value != 0);
    int *data = (int*)ndarray->data;
    for(int i = 0; i < ndarray->shape->len; i++) {
        data[i] %= value;
    }
}
