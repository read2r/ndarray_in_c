#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "ndarray.h"
#include "ndshape.h"

size_t get_item_size(DataType datatype) {
    switch(datatype) {
    case DT_INT:
        return sizeof(int);
        break;
    case DT_DOUBLE:
        return sizeof(double);
        break;
    default:
        return -1;
    }
}

NdArray* NdArray_new(void *data, NdShape *ndshape, DataType datatype) {
    NdArray *ndarray = (NdArray*)malloc(sizeof(NdArray));
    ndarray->shape = NdShape_copy(ndshape);
    ndarray->datatype = datatype;
    ndarray->item_size = get_item_size(ndarray->datatype);
    ndarray->size = ndarray->item_size * ndarray->shape->len;
    ndarray->data = malloc(ndarray->size);
    if(data == NULL) {
        memset(ndarray->data, 0, ndarray->size);
    } else {
        memcpy(ndarray->data, data, ndarray->size);
    }
    return ndarray;
}

NdArray* NdArray_copy(NdArray *src) {
    NdArray *dest = NdArray_new(NULL, src->shape, src->datatype);
    dest->size = src->size;
    memcpy(dest->data, src->data, src->size);
    return dest;
}

void NdArray_free(NdArray **dptr_ndarray) {
    NdShape_free(&((*dptr_ndarray)->shape));
    free(*dptr_ndarray);
    *dptr_ndarray= NULL;
}

NdArray* NdArray_zeros(unsigned int len, DataType datatype) {
    NdArray *ndarray = (NdArray*)malloc(sizeof(NdArray));
    ndarray->shape = NdShape_new(1, len);
    ndarray->datatype = datatype;
    ndarray->item_size = get_item_size(ndarray->datatype);
    ndarray->size = ndarray->item_size * ndarray->shape->len;
    ndarray->data = malloc(ndarray->size);
    memset(ndarray->data, 0, ndarray->size);
    return ndarray;
}

NdArray* NdArray_ones(unsigned int len, DataType datatype) {
    NdArray *ndarray = NdArray_zeros(len, datatype);
    for(int i = 0; i < len; i++) {
        switch(datatype) {
        case DT_INT:
            *((int*)ndarray->data + i) = 1;
            break;
        case DT_DOUBLE:
            *((double*)ndarray->data + i) = 1.0;
            break;
        default:
            abort();
        }
    }
    return ndarray;
}

NdArray* NdArray_arange(unsigned int start, unsigned int end, DataType datatype) {
    NdArray *ndarray = NdArray_zeros(end - start, datatype);
    for(int i = 0; i < end - start; i++) {
        switch(datatype) {
        case DT_INT:
            *((int*)ndarray->data + i) = start + i;
            break;
        case DT_DOUBLE:
            *((double*)ndarray->data + i) = (double)(start + i);
            break;
        default:
            abort();
        }
    }
    return ndarray;
}

int NdArray_reshape(NdArray *ndarray, NdShape *ndshape) {
    return NdShape_reshape(ndarray->shape, ndshape);
}

void* NdArray_getAt(NdArray *ndarray, unsigned int *position) {
    unsigned int offset = 0;
    unsigned int len = ndarray->shape->len;

    for(int i = 0; i < ndarray->shape->dim; i++) {
        len /= ndarray->shape->arr[i];
        offset += position[i] * len;
    }
    offset *= ndarray->item_size;
    return ndarray->data + offset;
}

void NdArray_setAt(NdArray *ndarray, unsigned int *position, void* data) {
    void *target_address = NdArray_getAt(ndarray, position);

    switch(ndarray->datatype) {
    case DT_INT:
        *((int*)target_address) = *(int*)data;
        break;
    case DT_DOUBLE:
        *((double*)target_address) = *(double*)data;
        break;
    default:
        fprintf(stderr, "invalid datatype");
        abort();
    }
}

void print_element(NdArray *ndarray, unsigned int *position) {
    void* ptr_element = NdArray_getAt(ndarray, position);
    switch(ndarray->datatype) {
    case DT_INT:
        printf("%d ", *(int*)ptr_element);
        break;
    case DT_DOUBLE:
        printf("%f ", *(double*)ptr_element);
        break;
    default:
        break;
    }
}

void printArray(NdArray *ndarray, unsigned int *position, int dim) {
    if(dim == ndarray->shape->dim) {
        print_element(ndarray, position);
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
        void *ptr_data_dest = dest->data; 
        void *ptr_data_src = src->data;
        switch(dest->datatype) {
        case DT_INT:
            *((int*)ptr_data_dest) += *((int*)ptr_data_src);
            break;
        case DT_DOUBLE:
            *((double*)ptr_data_dest) += *((double*)ptr_data_src);
            break;
        default:
            abort();
        }
        ptr_data_dest += dest->item_size;
        ptr_data_src += src->item_size;
    }

    return 1;
}

int NdArray_sub(NdArray *dest, NdArray *src) {
    if(!NdShape_compare(dest->shape, src->shape)) {
        return 0;
    }

    for(int i = 0; i < dest->shape->len; i++) {
        void *ptr_data_dest = dest->data; 
        void *ptr_data_src = src->data;
        switch(dest->datatype) {
        case DT_INT:
            *((int*)ptr_data_dest) -= *((int*)ptr_data_src);
            break;
        case DT_DOUBLE:
            *((double*)ptr_data_dest) -= *((double*)ptr_data_src);
            break;
        default:
            abort();
        }
        ptr_data_dest += dest->item_size;
        ptr_data_src += src->item_size;
    }

    return 1;
}

int NdArray_mul(NdArray *dest, NdArray *src) {
    if(!NdShape_compare(dest->shape, src->shape)) {
        return 0;
    }

    for(int i = 0; i < dest->shape->len; i++) {
        void *ptr_data_dest = dest->data; 
        void *ptr_data_src = src->data;
        switch(dest->datatype) {
        case DT_INT:
            *((int*)ptr_data_dest) *= *((int*)ptr_data_src);
            break;
        case DT_DOUBLE:
            *((double*)ptr_data_dest) *= *((double*)ptr_data_src);
            break;
        default:
            abort();
        }
        ptr_data_dest += dest->item_size;
        ptr_data_src += src->item_size;
    }

    return 1;
}

int NdArray_div(NdArray *dest, NdArray *src) {
    if(!NdShape_compare(dest->shape, src->shape)) {
        return 0;
    }

    for(int i = 0; i < dest->shape->len; i++) {
        void *ptr_data_dest = dest->data; 
        void *ptr_data_src = src->data;
        switch(dest->datatype) {
        case DT_INT:
            *((int*)ptr_data_dest) /= *((int*)ptr_data_src);
            break;
        case DT_DOUBLE:
            *((double*)ptr_data_dest) /= *((double*)ptr_data_src);
            break;
        default:
            abort();
        }
        ptr_data_dest += dest->item_size;
        ptr_data_src += src->item_size;
    }

    return 1;
}

int NdArray_mod(NdArray *dest, NdArray *src) {
    if(!NdShape_compare(dest->shape, src->shape)) {
        return 0;
    }

    for(int i = 0; i < dest->shape->len; i++) {
        void *ptr_data_dest = dest->data; 
        void *ptr_data_src = src->data;
        switch(dest->datatype) {
        case DT_INT:
            *((int*)ptr_data_dest) %= *((int*)ptr_data_src);
            break;
        case DT_DOUBLE:
        default:
            abort();
        }
        ptr_data_dest += dest->item_size;
        ptr_data_src += src->item_size;
    }

    return 1;
}

void dot_recursive(NdArray *result, NdArray *a, NdArray *b, unsigned int *position, unsigned int dim) {
    NdShape *shape_result = result->shape;

    if(dim >= shape_result->dim) {
        NdShape *shape_a = a->shape;
        NdShape *shape_b = b->shape;

        unsigned int *position_a = (unsigned int*)malloc(sizeof(unsigned int) * shape_a->dim);
        unsigned int *position_b = (unsigned int*)malloc(sizeof(unsigned int) * shape_b->dim);

        memcpy(position_a, position, sizeof(unsigned int) * (shape_a->dim-1));
        memcpy(position_b, position + (shape_a->dim-1), sizeof(unsigned int) * (shape_b->dim-1));
        position_b[shape_b->dim-1] = position_b[shape_b->dim-2];

        void *ptr_value_a;
        void *ptr_value_b;
        void *ptr_value_result = NdArray_getAt(result, position);
        memset(ptr_value_result, 0, result->item_size);

        for(int i = 0; i < shape_a->arr[shape_a->dim-1]; i++) {
            position_a[shape_a->dim-1] = i;
            position_b[shape_b->dim-2] = i;

            ptr_value_a = NdArray_getAt(a, position_a);
            ptr_value_b = NdArray_getAt(b, position_b);
            
            if(result->datatype == DT_INT) {
                *(int*)ptr_value_result += (*(int*)ptr_value_a) * (*(int*)ptr_value_b);
            } else if(result->datatype == DT_DOUBLE) {
                *(double*)ptr_value_result += (*(double*)ptr_value_a) * (*(double*)ptr_value_b);
            }
        }

        return;
    }

    for(int i = 0; i < shape_result->arr[dim]; i++) {
        position[dim] = i;
        dot_recursive(result, a, b, position, dim+1);
    }
}

void matmul_recursive(NdArray *result, NdArray *a, NdArray *b, unsigned int *position, unsigned int dim) {
    NdShape *shape_result = result->shape;

    if(dim >= shape_result->dim) {
        NdShape *shape_a = a->shape;
        NdShape *shape_b = b->shape;

        unsigned int *position_a = (unsigned int*)malloc(sizeof(unsigned int) * shape_a->dim);
        unsigned int *position_b = (unsigned int*)malloc(sizeof(unsigned int) * shape_b->dim);
        
        int offset_a = (shape_a->dim >= shape_b->dim) ? 0 : shape_b->dim - shape_a->dim;
        int offset_b = (shape_a->dim >= shape_b->dim) ? shape_a->dim - shape_b->dim : 0;
        
        memcpy(position_a, position + offset_a, sizeof(unsigned int) * shape_a->dim);
        memcpy(position_b, position + offset_b, sizeof(unsigned int) * shape_b->dim);

        void *ptr_value_a;
        void *ptr_value_b;
        void *ptr_value_result = NdArray_getAt(result, position);
        memset(ptr_value_result, 0, result->item_size);

        for(int i = 0; i < shape_a->arr[shape_a->dim-1]; i++) {
            position_a[shape_a->dim-1] = i;
            position_b[shape_b->dim-2] = i;

            ptr_value_a = NdArray_getAt(a, position_a);
            ptr_value_b = NdArray_getAt(b, position_b);
            
            if(result->datatype == DT_INT) {
                *(int*)ptr_value_result += (*(int*)ptr_value_a) * (*(int*)ptr_value_b);
            } else if(result->datatype == DT_DOUBLE) {
                *(double*)ptr_value_result += (*(double*)ptr_value_a) * (*(double*)ptr_value_b);
            }
        }

        return;
    }

    for(int i = 0; i < shape_result->arr[dim]; i++) {
        position[dim] = i;
        matmul_recursive(result, a, b, position, dim+1);
    }
}


NdArray* NdArray_dot(NdArray *a, NdArray *b) {
    NdArray *result;
    NdShape *shape_result;
    DataType datatype_result;

    datatype_result = a->datatype;

    NdShape *shape_a = a->shape;
    NdShape *shape_b = b->shape;

    if(shape_a->arr[shape_a->dim-1] != shape_b->arr[shape_b->dim-2]) {
        return NULL;
    }

    shape_result = NdShape_empty(shape_a->dim + shape_b->dim - 2);
    memcpy(shape_result->arr, shape_a->arr, sizeof(unsigned int) * (shape_a->dim-1));
    memcpy(shape_result->arr + (shape_a->dim-1), shape_b->arr, sizeof(unsigned int) * (shape_b->dim-1));
    shape_result->arr[shape_result->dim-1] = shape_b->arr[shape_b->dim-1];

    for(int i = 0; i < shape_result->dim; i++) {
        shape_result->len *= shape_result->arr[i];
    }

    result = NdArray_new(NULL, shape_result, datatype_result);

    unsigned int *position = (unsigned int*)malloc(sizeof(unsigned int) * shape_result->dim);
    memset(position, 0, sizeof(unsigned int) * shape_result->dim);

    dot_recursive(result, a, b, position, 0);
    
    return result; 
}

NdArray* NdArray_matmul(NdArray *a, NdArray *b) {
    NdArray *result;
    NdShape *shape_result;
    DataType datatype_result;

    datatype_result = a->datatype;
    NdShape *shape_a = a->shape;
    NdShape *shape_b = b->shape;

    if(shape_a->arr[shape_a->dim-1] != shape_b->arr[shape_b->dim-2]) {
        printf("nope1\n");
        return NULL;
    }

    int bound = (shape_a->dim >= shape_b->dim) ? shape_b->dim-2 : shape_a->dim-2;
    for(int i = 0; i < bound; i++) {
        if(shape_a->arr[shape_a->dim-i-3] != shape_b->arr[shape_b->dim-i-3]) {
            return NULL;
        }
    }

    if(shape_a->dim >= shape_b->dim) {
        shape_result = NdShape_copy(shape_a);
        shape_result->arr[shape_result->dim-1] = shape_b->arr[shape_b->dim-1];
    } else {
        shape_result = NdShape_copy(shape_b);
        shape_result->arr[shape_result->dim-2] = shape_a->arr[shape_a->dim-2];
    }
    result = NdArray_new(NULL, shape_result, datatype_result);

    unsigned int *position = (unsigned int*)malloc(sizeof(unsigned int) * shape_result->dim);
    memset(position, 0, shape_result->dim);
    
    matmul_recursive(result, a, b, position, 0);

    return result;
}

void NdArray_add_scalar(NdArray *ndarray, double value) {
    void *ptr_data = ndarray->data;
    for(int i = 0; i < ndarray->shape->len; i++) {
        switch(ndarray->datatype) {
        case DT_INT:
            *(int*)ptr_data += (int)value;
            break;
        case DT_DOUBLE:
            *(double*)ptr_data += value;
            break;
        default:
            abort();
        }
        ptr_data += ndarray->item_size;
    }
}

void NdArray_sub_scalar(NdArray *ndarray, double value) {
    void *ptr_data = ndarray->data;
    for(int i = 0; i < ndarray->shape->len; i++) {
        switch(ndarray->datatype) {
        case DT_INT:
            *(int*)ptr_data -= (int)value;
            break;
        case DT_DOUBLE:
            *(double*)ptr_data -= value;
            break;
        default:
            abort();
        }
        ptr_data += ndarray->item_size;
    }
}

void NdArray_mul_scalar(NdArray *ndarray, double value) {
    void *ptr_data = ndarray->data;
    for(int i = 0; i < ndarray->shape->len; i++) {
        switch(ndarray->datatype) {
        case DT_INT:
            *(int*)ptr_data *= (int)value;
            break;
        case DT_DOUBLE:
            *(double*)ptr_data *= value;
            break;
        default:
            abort();
        }
        ptr_data += ndarray->item_size;
    }
}

void NdArray_div_scalar(NdArray *ndarray, double value) {
    assert(value != 0);
    void *ptr_data = ndarray->data;
    for(int i = 0; i < ndarray->shape->len; i++) {
        switch(ndarray->datatype) {
        case DT_INT:
            *(int*)ptr_data /= (int)value;
            break;
        case DT_DOUBLE:
            *(double*)ptr_data /= value;
            break;
        default:
            abort();
        }
        ptr_data += ndarray->item_size;
    }
}

void NdArray_mod_scalar(NdArray *ndarray, int value) {
    assert(value != 0);
    void *ptr_data = ndarray->data;
    for(int i = 0; i < ndarray->shape->len; i++) {
        switch(ndarray->datatype) {
        case DT_INT:
            *(int*)ptr_data %= (int)value;
            break;
        case DT_DOUBLE:
        default:
            abort();
        }
        ptr_data += ndarray->item_size;
    }
}
