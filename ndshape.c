#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "ndshape.h"

NdShape* NdShape_empty(unsigned int dim) {
    NdShape *ndshape = (NdShape*)malloc(sizeof(NdShape));
    ndshape->dim = dim;
    ndshape->len = 1;
    ndshape->arr = (unsigned int*)malloc(sizeof(unsigned int) * dim);
    memset(ndshape->arr, 0, sizeof(unsigned int) * dim);
    return ndshape;
}

NdShape* NdShape_set(NdShape *ndshape, unsigned int dim, ...) {
    if(ndshape->dim != dim) {
        NdShape_free(&ndshape);
        ndshape = NdShape_empty(dim);
    }
    va_list ap;
    va_start(ap, dim);
    for(int i = 0; i < dim; i++) {
        int temp = va_arg(ap, int);
        ndshape->arr[i] = temp;
        ndshape->len *= temp;
    }
    va_end(ap);
    return ndshape;
}

NdShape* NdShape_new(unsigned int dim, ...) {
    NdShape *ndshape = NdShape_empty(dim);
    va_list ap;
    va_start(ap, dim);
    for(int i = 0; i < dim; i++) {
        int temp = va_arg(ap, int);
        ndshape->arr[i] = temp;
        ndshape->len *= temp;
    }
    va_end(ap);
    return ndshape;
}

NdShape* NdShape_copy(const NdShape *src) {
    NdShape *dest = NdShape_empty(src->dim);
    for(int i = 0; i < src->dim; i++) {
        dest->arr[i] = src->arr[i];
        dest->len *= src->arr[i];
    }
    return dest;
}

void NdShape_free(NdShape **ptrshape) {
    free((*ptrshape)->arr);
    free(*ptrshape);
    (*ptrshape)->arr= NULL;
    *ptrshape= NULL;
}

void NdShape_print(NdShape *ndshape) {
    printf("( ");
    for(int i = 0; i < ndshape->dim; i++) {
        printf("%d", ndshape->arr[i]);
        if(i < ndshape->dim-1) {
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

    NdShape_free(&dest);
    dest = NdShape_copy(src);
    return 1;
}
