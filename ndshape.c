#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "ndshape.h"

unsigned int* _create_shape_arr(unsigned int dim) {
    unsigned int *arr = (unsigned int*)malloc(sizeof(unsigned int) * dim);
    return arr;
}

NdShape* _create_shape(unsigned int dim) {
    NdShape* ndshape = (NdShape*)malloc(sizeof(NdShape));
    ndshape->arr = _create_shape_arr(dim);
    return ndshape;
}

void _set_shape_dim(NdShape *self, unsigned int dim) {
    self->dim = dim;
}

void _set_shape_arr_zeros(NdShape* self) {
    memset(self->arr, 0, sizeof(unsigned int) * self->dim);
}

void _set_shape_arr_fixed_array(NdShape *self, unsigned int dim, unsigned int *arr) {
    memcpy(self->arr, arr, sizeof(unsigned int) * self->dim);
}

void _set_shape_arr_fixed_array_reverse(NdShape *self, unsigned int dim, unsigned int *arr) {
    unsigned int reversed_arr[dim];
    for(int i = 0; i < dim; i++) {
        reversed_arr[i] = arr[dim-i-1];
    }
    _set_shape_arr_fixed_array(self, dim, reversed_arr);
}

void _set_shape_arr_va_list(NdShape *self, unsigned int dim, va_list args) {
    for(int i = 0; i < dim; i++) {
        self->arr[i] = va_arg(args, unsigned int);
    }
}

void _set_shape_len(NdShape *self) {
    self->len = 1;
    for(int i = 0; i < self->dim; i++) {
        self->len *= self->arr[i];
    }
    self->len = (self->len != 0) ? self->len : 1;
}

NdShape* NdShape_empty(unsigned int dim) {
    NdShape *ndshape = _create_shape(dim);
    _set_shape_dim(ndshape, dim);
    _set_shape_arr_zeros(ndshape);
    _set_shape_len(ndshape);
    return ndshape;
}

NdShape* NdShape_set_fixed_array(NdShape *self, unsigned int dim, unsigned int *arr) {
    _set_shape_dim(self, dim);
    _set_shape_arr_fixed_array(self, dim, arr);
    _set_shape_len(self);
    return self;
}

NdShape* NdShape_set_va_list(NdShape *self, unsigned int dim, va_list args) {
    _set_shape_dim(self, dim);
    _set_shape_arr_va_list(self, dim, args);
    _set_shape_len(self);
    return self;
}

NdShape* NdShape_set(NdShape *self, unsigned int dim, ...) {
    va_list args;
    va_start(args, dim);
    NdShape_set_va_list(self, dim, args);
    va_end(args);
    return self;
}

NdShape* NdShape_new_fixed_array(unsigned int dim, unsigned int *arr) {
    NdShape *ndshape = NdShape_empty(dim);
    _set_shape_arr_fixed_array(ndshape, dim, arr);
    _set_shape_len(ndshape);
    return ndshape;
}

NdShape* NdShape_new(unsigned int dim, ...) {
    NdShape *ndshape = NdShape_empty(dim);
    va_list args;
    va_start(args, dim);
    _set_shape_arr_va_list(ndshape, dim, args);
    _set_shape_len(ndshape);
    va_end(args);
    return ndshape;
}

NdShape* NdShape_copy(const NdShape *src) {
    NdShape *dest = NdShape_empty(src->dim);
    _set_shape_arr_fixed_array(dest, src->dim, src->arr);
    _set_shape_len(dest);
    return dest;
}

void NdShape_free(NdShape **ptr_shape) {
    free((*ptr_shape)->arr);
    free(*ptr_shape);
    (*ptr_shape)->arr= NULL;
    *ptr_shape= NULL;
}

void NdShape_print(NdShape *self) {
    printf("( ");
    for(int i = 0; i < self->dim; i++) {
        printf("%d", self->arr[i]);
        if(i < self->dim-1) {
            printf(", ");
        }
    }
    printf(" )\n");
}

int NdShape_compare(NdShape *a, NdShape *b) {
    if(a->dim != b->dim) {
        return 0;
    }

    if(a->len != b ->len) {
        return 0;
    }

    for(int i = 0; i < a->dim; i++) {
        if(a->arr[i] != b->arr[i]) {
            return 0;
        }
    }

    return 1;
}

int NdShape_reshape(NdShape *dest, const NdShape *src) {
    if(dest->len != src->len) {
        fprintf(stderr, "lengthes of both must be same. (%d %d)\n", dest->len, src->len);
        return 0;
    }
    _set_shape_dim(dest, src->dim);
    _set_shape_arr_fixed_array(dest, src->dim, src->arr);
    return 1;
}

NdShape* NdShape_reverse(NdShape *self) {
    NdShape *reversed = NdShape_copy(self);
    _set_shape_arr_fixed_array_reverse(reversed, self->dim, self->arr);
    return reversed;
}
