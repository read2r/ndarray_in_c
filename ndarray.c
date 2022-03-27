#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <omp.h>
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
    case DT_BOOL:
        return sizeof(char);
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
    memcpy(dest->data, src->data, src->size);
    return dest;
}

void NdArray_free(NdArray **dptr_ndarray) {
    NdShape_free(&((*dptr_ndarray)->shape));
    free((*dptr_ndarray)->data);
    free(*dptr_ndarray);
    *dptr_ndarray= NULL;
}

void NdArray_sub_free(NdArray **dptr_ndarray) {
    NdShape_free(&((*dptr_ndarray)->shape));
    free(*dptr_ndarray);
    *dptr_ndarray= NULL;
}

NdArray* NdArray_zeros(unsigned int len, DataType datatype) {
    NdShape *ndshape = NdShape_new(1, len);
    NdArray *ndarray = NdArray_new(NULL, ndshape, datatype);
    NdShape_free(&ndshape);
    return ndarray;
}

NdArray* NdArray_ones(unsigned int len, DataType datatype) {
    NdArray *ndarray = NdArray_zeros(len, datatype);
    void *cur = ndarray->data;
    for(int i = 0; i < len; i++) {
        switch(datatype) {
        case DT_INT:
            *((int*)cur + i) = 1;
            break;
        case DT_DOUBLE:
            *((double*)cur + i) = 1.0;
            break;
        default:
            abort();
        }
    }
    return ndarray;
}

NdArray* NdArray_arange(unsigned int start, unsigned int end, DataType datatype) {
    int len = end - start;
    NdArray *ndarray = NdArray_zeros(len, datatype);
    void *cur = ndarray->data;
    for(int i = 0; i < len; i++) {
        switch(datatype) {
        case DT_INT:
            *((int*)cur + i) = start + i;
            break;
        case DT_DOUBLE:
            *((double*)cur + i) = (double)(start + i);
            break;
        default:
            abort();
        }
    }
    return ndarray;
}

NdArray* NdArray_random(unsigned int len, DataType datatype) {
    NdArray *ndarray = NdArray_zeros(len, datatype);
    void* cur = ndarray->data;
    srand(time(NULL));
    for(int i = 0; i < ndarray->shape->len; i++) {
        switch(datatype) {
        case DT_INT:
            *((int*)cur + i) = rand();
            break;
        case DT_DOUBLE:
            *((double*)cur + i) = (double)rand() / (RAND_MAX - 10);
            break;
        default:
            abort();
        }
    }
    return ndarray;
}

NdArray* NdArray_random_range(unsigned int len, unsigned int low, unsigned int high, DataType datatype) {
    assert(len > 0);
    assert(low < high);

    NdArray *ndarray = NdArray_zeros(len, datatype);
    void* cur = ndarray->data;
    int bound = high - low;
    srand(time(NULL));
    for(int i = 0; i < ndarray->shape->len; i++) {
        switch(datatype) {
        case DT_INT:
            *((int*)cur + i) = rand() % bound + low;
            break;
        case DT_DOUBLE:
            *((double*)cur + i) = (double)rand() / (RAND_MAX) + rand() % bound + low;
            break;
        default:
            abort();
        }
    }
    return ndarray;
}

double get_gaussian_random_value() {
    double v1, v2, s;
    do {
        v1 = 2 * ((double) rand() / RAND_MAX) - 1;
        v2 = 2 * ((double) rand() / RAND_MAX) - 1;
        s = v1 * v1 + v2 * v2;
    } while(s >= 1 || s == 0);

    s = sqrt((-2 * log(s)) / s);

    return v1 * s;
}

NdArray* NdArray_random_gaussian(unsigned int len) {
    NdArray *ndarray = NdArray_zeros(len, DT_DOUBLE);
    double *cur = ndarray->data;
    srand(time(NULL));
    for(int i = 0; i < len; i++) {
        cur[i] = get_gaussian_random_value();
    }
    return ndarray;
}

NdArray* NdArray_shuffle(NdArray *array) {
    srand(time(NULL));
    for(int i = array->shape->len-1; i > 0; i--) {
        int random_idx = rand() % i;
        switch(array->datatype) {
        case DT_INT:
            {
                int *cur = array->data;
                int temp = cur[i];
                cur[i] = cur[random_idx];
                cur[random_idx] = temp;
            }
            break;
        case DT_DOUBLE:
            {
                double *cur = array->data;
                double temp = cur[i];
                cur[i] = cur[random_idx];
                cur[random_idx] = temp;
            }
            break;
        default:
            abort();
        }
    }
    return array;
}

NdArray* NdArray_choice(unsigned int pick_len, unsigned int len, DataType datatype) {
    assert(len >= pick_len );
    NdArray* choices = NdArray_zeros(pick_len, datatype);
    NdArray* temp = NdArray_arange(0, len, datatype);
    NdArray_shuffle(temp);
    memcpy(choices->data, temp->data, choices->size);
    NdArray_free(&temp);
    return choices;
}

int NdArray_reshape(NdArray *ndarray, NdShape *ndshape) {
    return NdShape_reshape(ndarray->shape, ndshape);
}

int NdArray_reshape_fixed_array(NdArray *self, unsigned int dim, unsigned int *arr) {
    return NdShape_reshape_fixed_array(self->shape, dim, arr);
}

int NdArray_reshape_variadic(NdArray *self, unsigned int dim, ...) {
    unsigned int arr[dim];
    va_list args;
    va_start(args, dim);
    for(int i = 0; i < dim; i++) {
        arr[i] = va_arg(args, unsigned int);
    }
    va_end(args);
    return NdShape_reshape_fixed_array(self->shape, dim, arr);
}

unsigned int get_offset(NdArray *ndarray, unsigned int *position, int pdim) {
    unsigned int offset = 0;
    unsigned int len = ndarray->shape->len;
    for(int i = 0; i < pdim; i++) {
        len /= ndarray->shape->arr[i];
        offset += position[i] * len;
    }
    offset *= ndarray->item_size;
    return offset;
}

void* NdArray_getAt(NdArray *ndarray, unsigned int *position) {
    unsigned int offset = get_offset(ndarray, position, ndarray->shape->dim);
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

NdArray* NdArray_subarray(NdArray *ndarray, unsigned int *position, int pdim) {
    unsigned int offset = get_offset(ndarray, position, pdim);
    unsigned int subarray_dim = ndarray->shape->dim - pdim;
    NdShape *subshape = NdShape_empty(subarray_dim);
    for(int i = 0; i < subarray_dim; i++) {
        subshape->arr[i] = ndarray->shape->arr[pdim + i];
        subshape->len *= subshape->arr[i];
    }
    NdArray *subarray = NdArray_new(ndarray->data + offset, subshape, ndarray->datatype);
    return subarray;
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
    case DT_BOOL:
        printf("%d ", *(char*)ptr_element);
        break;
    default:
        abort();
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
    unsigned int position[ndarray->shape->dim];
    memset(position, 0, sizeof(unsigned int) * ndarray->shape->dim);
    printArray(ndarray, position, 0);
    printf("\n");
}

void NdArray_printShape(NdArray *ndarray) {
    return NdShape_print(ndarray->shape);
}

int NdArray_is_arithmetically_vaild(NdArray *a, NdArray *b){
    if(a->datatype != b->datatype) {
        return 0;
    }

    if(a->shape->dim < b->shape->dim) {
        return 0;
    }

    for(int i = 0; i < b->shape->dim; i++) {
        if(a->shape->arr[a->shape->dim-i-1] != b->shape->arr[b->shape->dim-i-1]) {
            return 0;
        }
    }

    return 1;
}

int NdArray_add(NdArray *dest, NdArray *src) {
    void *cur_dest;
    void *cur_src;

    if(!NdArray_is_arithmetically_vaild(dest, src)) {
        return 0;
    }

    cur_dest = dest->data;
    for(int i = 0; i < dest->shape->len / src->shape->len; i++) {
        cur_src = src->data;
        for(int j = 0; j < src->shape->len; j++) {
            switch(dest->datatype) {
            case DT_INT:
                *((int*)cur_dest) += *((int*)cur_src);
                break;
            case DT_DOUBLE:
                *((double*)cur_dest) += *((double*)cur_src);
                break;
            default:
                abort();
            }
            cur_dest += dest->item_size;
            cur_src += src->item_size;
        }
    }

    return 1;
}

int NdArray_sub(NdArray *dest, NdArray *src) {
    void *cur_dest;
    void *cur_src;

    if(!NdArray_is_arithmetically_vaild(dest, src)) {
        return 0;
    }

    cur_dest = dest->data;
    for(int i = 0; i < dest->shape->len / src->shape->len; i++) {
        cur_src = src->data;
        for(int j = 0; j < src->shape->len; j++) {
            switch(dest->datatype) {
            case DT_INT:
                *((int*)cur_dest) -= *((int*)cur_src);
                break;
            case DT_DOUBLE:
                *((double*)cur_dest) -= *((double*)cur_src);
                break;
            default:
                abort();
            }
            cur_dest += dest->item_size;
            cur_src += src->item_size;
        }
    }

    return 1;
}

int NdArray_mul(NdArray *dest, NdArray *src) {
    void *cur_dest;
    void *cur_src;

    if(!NdArray_is_arithmetically_vaild(dest, src)) {
        return 0;
    }

    cur_dest = dest->data;
    for(int i = 0; i < dest->shape->len / src->shape->len; i++) {
        cur_src = src->data;
        for(int j = 0; j < src->shape->len; j++) {
            switch(dest->datatype) {
            case DT_INT:
                *((int*)cur_dest) *= *((int*)cur_src);
                break;
            case DT_DOUBLE:
                *((double*)cur_dest) *= *((double*)cur_src);
                break;
            default:
                abort();
            }
            cur_dest += dest->item_size;
            cur_src += src->item_size;
        }
    }

    return 1;
}

int NdArray_div(NdArray *dest, NdArray *src) {
    void *cur_dest;
    void *cur_src;

    if(!NdArray_is_arithmetically_vaild(dest, src)) {
        return 0;
    }

    cur_dest = dest->data;
    for(int i = 0; i < dest->shape->len / src->shape->len; i++) {
        cur_src = src->data;
        for(int j = 0; j < src->shape->len; j++) {
            switch(dest->datatype) {
            case DT_INT:
                *((int*)cur_dest) /= *((int*)cur_src);
                break;
            case DT_DOUBLE:
                *((double*)cur_dest) /= *((double*)cur_src);
                break;
            default:
                abort();
            }
            cur_dest += dest->item_size;
            cur_src += src->item_size;
        }
    }

    return 1;
}

int NdArray_mod(NdArray *dest, NdArray *src) {
    assert(dest->datatype != DT_DOUBLE || src->datatype != DT_DOUBLE);

    if(!NdShape_compare(dest->shape, src->shape)) {
        return 0;
    }

    void *ptr_data_dest = dest->data; 
    void *ptr_data_src = src->data;
    for(int i = 0; i < dest->shape->len; i++) {
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

void matmul_int(int *a, int *b, int *c, int p, int q, int r) {
    #pragma omp parallel for
    for(int i = 0; i < p; i++) {
        for(int k = 0; k < q; k++) {
            register int v = a[i * q + k];
            for(int j = 0; j < r; j++) {
                c[i * r + j] += v * b[k * r + j];
            }
        }
    }
}

void matmul_double(double *a, double *b, double *c, int p, int q, int r) {
    #pragma omp parallel for
    for(int i = 0; i < p; i++) {
        for(int k = 0; k < q; k++) {
            register double v = a[i * q + k];
            for(int j = 0; j < r; j++) {
                c[i * r + j] += v * b[k * r + j];
            }
        }
    }
}

NdArray* matmul_2d(NdArray *a, NdArray *b) {
    NdShape *shape_result = NdShape_new(2, a->shape->arr[0], b->shape->arr[1]);
    NdArray *result = NdArray_new(NULL, shape_result, a->datatype);
    switch(result->datatype) {
    case DT_INT:
        matmul_int(a->data, b->data, result->data,
            result->shape->arr[0], a->shape->arr[1], result->shape->arr[1]);
        break;
    case DT_DOUBLE:
        matmul_double(a->data, b->data, result->data,
            result->shape->arr[0], a->shape->arr[1], result->shape->arr[1]);
        break;
    default:
        abort();
    }
    return result;
}

void matmul_recursive(NdArray *result, NdArray *a, NdArray *b, unsigned int *position, unsigned int dim) {
    NdShape *shape_result = result->shape;

    if(dim >= shape_result->dim-2) {
        NdShape *shape_a = a->shape;
        NdShape *shape_b = b->shape;

        unsigned int position_a[shape_a->dim-2];
        unsigned int position_b[shape_b->dim-2];
        
        int offset_a = (shape_a->dim >= shape_b->dim) ? 0 : shape_b->dim - shape_a->dim;
        int offset_b = (shape_a->dim >= shape_b->dim) ? shape_a->dim - shape_b->dim : 0;
        
        memcpy(position_a, position + offset_a, sizeof(unsigned int) * (shape_a->dim-2));
        memcpy(position_b, position + offset_b, sizeof(unsigned int) * (shape_b->dim-2));

        void *mat_a = a->data + get_offset(a, position_a, shape_a->dim-2);
        void *mat_b = b->data + get_offset(b, position_b, shape_b->dim-2);
        void *mat_result = result->data + get_offset(result, position, dim);

        int p = a->shape->arr[shape_a->dim-2];
        int q = a->shape->arr[shape_a->dim-1];
        int r = b->shape->arr[shape_b->dim-1];

        switch(result->datatype) {
        case DT_INT:
            matmul_int(mat_a, mat_b, mat_result, p, q, r);
            break;
        case DT_DOUBLE:
            matmul_double(mat_a, mat_b, mat_result, p, q, r);
            break;
        default:
            abort();
        }
        
        return;
    }

    for(int i = 0; i < shape_result->arr[dim]; i++) {
        position[dim] = i;
        matmul_recursive(result, a, b, position, dim+1);
    }
}

NdArray* matmul_nd(NdArray *a, NdArray *b) {
    NdArray *result;
    NdShape *shape_result;
    DataType datatype_result;

    datatype_result = a->datatype;
    NdShape *shape_a = a->shape;
    NdShape *shape_b = b->shape;

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

    unsigned int position[shape_result->dim-2];
    memset(position, 0, sizeof(unsigned int) * (shape_result->dim-2));

    matmul_recursive(result, a, b, position, 0);

    return result;
}

NdArray* dot_2d(NdArray *a, NdArray *b) {
    return matmul_2d(a, b);
}

void dot_recursive(NdArray *result, NdArray *a, NdArray *b, unsigned int *position, unsigned int dim) {
    NdShape *shape_result = result->shape;

    if(dim >= shape_result->dim) {
        NdShape *shape_a = a->shape;
        NdShape *shape_b = b->shape;

        unsigned int position_a[shape_a->dim];
        unsigned int position_b[shape_b->dim];

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

NdArray* dot_nd(NdArray *a, NdArray *b) {
    NdArray *result;
    NdShape *shape_result;
    DataType datatype_result;

    datatype_result = a->datatype;

    NdShape *shape_a = a->shape;
    NdShape *shape_b = b->shape;

    shape_result = NdShape_empty(shape_a->dim + shape_b->dim - 2);
    memcpy(shape_result->arr, shape_a->arr, sizeof(unsigned int) * (shape_a->dim-1));
    memcpy(shape_result->arr + (shape_a->dim-1), shape_b->arr, sizeof(unsigned int) * (shape_b->dim-1));
    shape_result->arr[shape_result->dim-1] = shape_b->arr[shape_b->dim-1];

    for(int i = 0; i < shape_result->dim; i++) {
        shape_result->len *= shape_result->arr[i];
    }

    result = NdArray_new(NULL, shape_result, datatype_result);

    unsigned int position[shape_result->dim];
    memset(position, 0, sizeof(unsigned int) * shape_result->dim);

    dot_recursive(result, a, b, position, 0);
    
    return result; 
}


NdArray* NdArray_dot(NdArray *a, NdArray *b) {
    if(a->shape->arr[a->shape->dim-1] != b->shape->arr[b->shape->dim-2]) {
        abort();
    }
    if(a->shape->dim == 2 && b->shape->dim == 2) {
        return dot_2d(a, b);
    } else {
        return dot_nd(a, b);
    }
}


NdArray* NdArray_matmul(NdArray *a, NdArray *b) {
    if(a->shape->arr[a->shape->dim-1] != b->shape->arr[b->shape->dim-2]) {
        abort();
    }
    if(a->shape->dim == 2 && b->shape->dim == 2) {
        return matmul_2d(a, b);
    } else {
        return matmul_nd(a, b);
    }
}

void transpose_recursive(NdArray *array, NdArray *transposed, unsigned int *position, int dim) {
    if(dim >= array->shape->dim) {
        unsigned int tdim = transposed->shape->dim;
        unsigned int position_transpose[tdim];
        for(int i = 0; i < tdim; i++) {
            position_transpose[i] = position[tdim-i-1];
        }

        unsigned int offset_array = get_offset(array, position, array->shape->dim);
        unsigned int offset_transposed = get_offset(transposed, position_transpose, transposed->shape->dim);

        void *cur_array = array->data + offset_array;
        void *cur_transposed = transposed->data + offset_transposed;

        memcpy(cur_transposed, cur_array, array->item_size);

        return;
    }

    for(int i = 0; i < array->shape->arr[dim]; i++) {
        position[dim] = i;
        transpose_recursive(array, transposed, position, dim+1);
    }
}

NdArray* NdArray_transpose(NdArray *self) {
    NdShape *shape_transposed = NdShape_reverse(self->shape);
    NdArray *transposed = NdArray_zeros(self->shape->len, self->datatype);
    NdArray_reshape(transposed, shape_transposed);

    unsigned int position[self->shape->dim];
    transpose_recursive(self, transposed, position, 0);

    NdShape_free(&shape_transposed);

    return transposed;
}

void transpose_axis_recursive(NdArray *array, NdArray *transposed, unsigned int *position, unsigned int *args, int dim) {
    if(dim >= array->shape->dim) {
        unsigned int tdim = transposed->shape->dim;
        unsigned int position_transpose[tdim];

        for(int i = 0; i < tdim; i++) {
            position_transpose[i] = position[args[i]];
        }

        unsigned int offset_array = get_offset(array, position, array->shape->dim);
        unsigned int offset_transposed = get_offset(transposed, position_transpose, transposed->shape->dim);

        void *cur_array = array->data + offset_array;
        void *cur_transposed = transposed->data + offset_transposed;

        memcpy(cur_transposed, cur_array, array->item_size);

        return;
    }

    for(int i = 0; i < array->shape->arr[dim]; i++) {
        position[dim] = i;
        transpose_axis_recursive(array, transposed, position, args, dim+1);
    }
}

NdArray* NdArray_transpose_axis(NdArray *self, int dim, ...) {
    unsigned int arr[dim];

    va_list args;
    va_start(args, dim);
    for(int i = 0; i < dim; i++) {
        arr[i] = va_arg(args, int);
    }
    va_end(args);

    unsigned int transposed_shape_data[dim];
    for(int i = 0; i < dim; i++) {
        transposed_shape_data[i] = self->shape->arr[arr[i]];
    }

    NdArray *transposed = NdArray_zeros(self->shape->len, self->datatype);
    NdArray_reshape_fixed_array(transposed, dim, transposed_shape_data);

    unsigned int position[self->shape->dim];
    transpose_axis_recursive(self, transposed, position, arr, 0);

    return transposed;
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
    assert(ndarray->datatype != DT_DOUBLE);
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

void NdArray_broadcast(NdArray *ndarray, broadcast_func bfunc) {
    void *ptr_data = ndarray->data;
    for(int i = 0; i < ndarray->shape->len; i++) {
        switch(ndarray->datatype) {
        case DT_INT:
        case DT_DOUBLE:
            bfunc(ptr_data);
            break;
        default:
            abort();
        }
        ptr_data += ndarray->item_size;
    }
}

long NdArray_sum_char(NdArray *ndarray) {
    char *data = (char*)ndarray->data;
    long sum = 0;
    for(int i = 0; i < ndarray->shape->len; i++) {
        sum += data[i];
    }
    return sum;
}

long NdArray_sum_int(NdArray *ndarray) {
    int *data = (int*)ndarray->data;
    long sum = 0;
    for(int i = 0; i < ndarray->shape->len; i++) {
        sum += data[i];
    }
    return sum;
}

long double NdArray_sum_double(NdArray *ndarray) {
    double *data = (double*)ndarray->data;
    long double sum = 0;
    for(int i = 0; i < ndarray->shape->len; i++) {
        sum += data[i];
    }
    return sum;
}

void* NdArray_sum(NdArray *ndarray) {
    void *ptr_sum = malloc(ndarray->item_size);
    if(ndarray->datatype == DT_INT) {
        *(int*)ptr_sum = NdArray_sum_int(ndarray);
    } else if(ndarray->datatype == DT_DOUBLE) {
        *(double*)ptr_sum = NdArray_sum_double(ndarray);
    } else if(ndarray->datatype == DT_BOOL) {
        *(char*)ptr_sum = NdArray_sum_char(ndarray);
    } else {
        abort();
    }
    return ptr_sum;
}

NdShape* get_ndshape_axis(NdShape *self, unsigned int axis) {
    NdShape *shape_sum = NdShape_empty(self->dim-1);
    for(int i = 0; i < shape_sum->dim; i++) {
        shape_sum->arr[i] = (i >= axis) ? self->arr[i+1] : self->arr[i];
        shape_sum->len *= shape_sum->arr[i];
    }
    return shape_sum;
}

NdArray* get_ndarray_axis(NdArray *self, unsigned int axis, DataType datatype) {
    NdShape *shape_self = self->shape;
    NdShape *shape_sum = get_ndshape_axis(self->shape, axis);
    NdArray *array_sum = NdArray_new(NULL, shape_sum, datatype);
    return array_sum;
}

unsigned int get_step_axis(NdShape *self, unsigned int axis) {
    unsigned int step = self->len;
    for(int i = 0; i <= axis; i++) {
        step /= self->arr[i];
    }
    return step;
}

NdArray* cal_array_sum_axis(NdArray *self, NdArray *result, unsigned int axis) {
    unsigned int step = get_step_axis(self->shape, axis);
    unsigned int memo[self->shape->len];
    for(int i = 0; i < self->shape->len; i++) {
        memo[i] = 0;
    }

    void *cur = self->data;
    void *cur_result = result->data;
    void *sum = malloc(result->item_size);

    for(int i = 0; i < self->shape->len; i++) {
        if(memo[i] == 1) {
            continue;
        }

        memset(sum, 0, result->item_size);
        for(int j = 0; j < self->shape->arr[axis]; j++) {
            int idx = i + j * step;
            switch(result->datatype) {
            case DT_INT:
                *(int*)sum += *((int*)cur + idx);
                break;
            case DT_DOUBLE:
                *(double*)sum += *((double*)cur + idx);
                break;
            default:
                abort();
            }
            memo[idx] = 1;
        }
        memcpy(cur_result, sum, result->item_size);
        cur_result += result->item_size;
    }

    free(sum);
    return result;
}

NdArray* cal_array_argmax_axis(NdArray *self, NdArray *result, unsigned int axis) {
    unsigned int step = get_step_axis(self->shape, axis);
    unsigned int memo[self->shape->len];
    for(int i = 0; i < self->shape->len; i++) {
        memo[i] = 0;
    }

    void *cur = self->data;
    int *cur_result = result->data;

    void *max = malloc(self->item_size);
    int max_idx;

    for(int i = 0; i < self->shape->len; i++) {
        if(memo[i] == 1) {
            continue;
        }

        memcpy(max, cur + i * self->item_size, self->item_size);
        max_idx = 0;
        for(int j = 1; j < self->shape->arr[axis]; j++) {
            int idx = i + j * step;
            switch(self->datatype) {
            case DT_INT:
                if(*(int*)max < *((int*)cur + idx)) {
                    *(int*)max = *((int*)cur + idx);
                    max_idx = j;
                }
                break;
            case DT_DOUBLE:
                if(*(double*)max < *((double*)cur + idx)) {
                    *(double*)max = *((double*)cur + idx);
                    max_idx = j;
                }
                break;
            default:
                abort();
            }
            memo[idx] = 1;
        }
        *cur_result = max_idx;
        cur_result++;
    }

    return result;
}

NdArray* cal_array_max_axis(NdArray *self, NdArray *result, unsigned int axis) {
    unsigned int step = get_step_axis(self->shape, axis);
    unsigned int memo[self->shape->len];
    for(int i = 0; i < self->shape->len; i++) {
        memo[i] = 0;
    }

    void *cur = self->data;
    void *cur_result = result->data;
    void *max = malloc(self->item_size);

    for(int i = 0; i < self->shape->len; i++) {
        if(memo[i] == 1) {
            continue;
        }

        memcpy(max, cur + i * self->item_size, self->item_size);
        for(int j = 1; j < self->shape->arr[axis]; j++) {
            int idx = i + j * step;
            switch(self->datatype) {
            case DT_INT:
                if(*(int*)max < *((int*)cur + idx)) {
                    *(int*)max = *((int*)cur + idx);
                }
                break;
            case DT_DOUBLE:
                if(*(double*)max < *((double*)cur + idx)) {
                    *(double*)max = *((double*)cur + idx);
                }
                break;
            default:
                abort();
            }
            memo[idx] = 1;
        }
        memcpy(cur_result, max, result->item_size);
        cur_result += result->item_size;
    }

    return result;
}

NdArray* NdArray_sum_axis(NdArray *self, unsigned int axis) {
    if(axis > self->shape->dim) {
        return NULL;
    }
    NdArray *array_sum = get_ndarray_axis(self, axis, self->datatype);
    cal_array_sum_axis(self, array_sum, axis);
    return array_sum;
}

NdArray* NdArray_max_axis(NdArray *self, unsigned int axis) {
    if(axis > self->shape->dim) {
        return NULL;
    }
    NdArray *array_max = get_ndarray_axis(self, axis, self->datatype);
    cal_array_max_axis(self, array_max, axis);
    return array_max;
}

NdArray* NdArray_argmax_axis(NdArray *self, unsigned int axis) {
    if(axis > self->shape->dim) {
        return NULL;
    }
    NdArray *array_argmax = get_ndarray_axis(self, axis, DT_INT);
    cal_array_argmax_axis(self, array_argmax, axis);
    return array_argmax;
}

int NdArray_max_int(NdArray *ndarray) {
    int *data = (int*)ndarray->data;
    int max = data[0];
    for(int i = 1; i < ndarray->shape->len; i++) {
        max = (max < data[i]) ? data[i] : max;
    }
    return max;
}
double NdArray_max_double(NdArray *ndarray) {
    double *data = (double*)ndarray->data;
    double max = data[0];
    for(int i = 1; i < ndarray->shape->len; i++) {
        max = (max < data[i]) ? data[i] : max;
    }
    return max;
}
void* NdArray_max(NdArray *ndarray) {
    void *ptr_max = malloc(ndarray->item_size);
    if(ndarray->datatype == DT_INT) {
        *(int*)ptr_max = NdArray_max_int(ndarray);
    } else if(ndarray->datatype == DT_DOUBLE) {
        *(double*)ptr_max = NdArray_max_double(ndarray);
    }
    return ptr_max;
}

int NdArray_min_int(NdArray *ndarray) {
    int *data = (int*)ndarray->data;
    int min = data[0];
    for(int i = 1; i < ndarray->shape->len; i++) {
        min = (min > data[i]) ? data[i] : min;
    }
    return min;
}

double NdArray_min_double(NdArray *ndarray) {
    double *data = (double*)ndarray->data;
    double min = data[0];
    for(int i = 1; i < ndarray->shape->len; i++) {
        min = (min > data[i]) ? data[i] : min;
    }
    return min;
}

void* NdArray_min(NdArray *ndarray) {
    void *ptr_min = malloc(ndarray->item_size);
    if(ndarray->datatype == DT_INT) {
        *(int*)ptr_min = NdArray_min_int(ndarray);
    } else if(ndarray->datatype == DT_DOUBLE) {
        *(double*)ptr_min = NdArray_min_double(ndarray);
    }
    return ptr_min;
}

double NdArray_mean_int(NdArray *ndarray) {
    int *data = (int*)ndarray->data;
    double mean = (double)NdArray_sum_int(ndarray) / ndarray->shape->len;
    return mean;
}

double NdArray_mean_double(NdArray *ndarray) {
    double *data = (double*)ndarray->data;
    double mean = NdArray_sum_double(ndarray) / ndarray->shape->len;
    return mean;
}

void* NdArray_mean(NdArray *ndarray) {
    void *ptr_mean = malloc(ndarray->item_size);
    if(ndarray->datatype == DT_INT) {
        *(int*)ptr_mean = NdArray_mean_int(ndarray);
    } else if(ndarray->datatype == DT_DOUBLE) {
        *(double*)ptr_mean = NdArray_mean_double(ndarray);
    }
    return ptr_mean;
}

int NdArray_argmax_int(NdArray *ndarray) {
    int *cur = ndarray->data;
    int max = *cur;
    unsigned int idx_max = 0;
    for(int i = 0; i < ndarray->shape->len; i++) {
        if(max < *cur) {
            max = *cur;
            idx_max = i;
        }
        cur++;
    }
    return idx_max;
}

int NdArray_argmax_double(NdArray *ndarray) {
    double *cur = ndarray->data;
    double  max = *cur;
    unsigned int idx_max = 0;
    for(int i = 0; i < ndarray->shape->len; i++) {
        if(max < *cur) {
            max = *cur;
            idx_max = i;
        }
        cur++;
    }
    return idx_max;
}

int NdArray_argmax(NdArray *ndarray) {
    void *cur = ndarray->data;
    if(ndarray->datatype == DT_INT) {
        return NdArray_argmax_int(ndarray);
    } else if(ndarray->datatype == DT_DOUBLE) {
        return NdArray_argmax_double(ndarray);
    }
    return -1;
}

int NdArray_argmin_int(NdArray *ndarray) {
    int *cur = ndarray->data;
    int min = *cur;
    unsigned int idx_min = 0;
    for(int i = 0; i < ndarray->shape->len; i++) {
        if(min > *cur) {
            min = *cur;
            idx_min = i;
        }
        cur++;
    }
    return idx_min;
}

int NdArray_argmin_double(NdArray *ndarray) {
    double *cur = ndarray->data;
    double min = *cur;
    unsigned int idx_min = 0;
    for(int i = 0; i < ndarray->shape->len; i++) {
        if(min > *cur) {
            min = *cur;
            idx_min = i;
        }
        cur++;
    }
    return idx_min;
}

int NdArray_argmin(NdArray *ndarray) {
    void *cur = ndarray->data;
    if(ndarray->datatype == DT_INT) {
        return NdArray_argmin_int(ndarray);
    } else if(ndarray->datatype == DT_DOUBLE) {
        return NdArray_argmin_double(ndarray);
    }
    return -1;
}

void _convert_datatype_from_int_to_double(NdArray **ptr_self) {
    NdArray *self = *ptr_self;
    NdArray *converted = NdArray_new(NULL, self->shape, DT_DOUBLE);

    int* cur_self = self->data;
    double* cur_conv = converted->data;

    for(int i = 0; i < converted->shape->len; i++) {
        cur_conv[i] = (double)cur_self[i];
    }

    NdArray_free(&self);
    *ptr_self = converted;
}

void _convert_datatype_from_double_to_int(NdArray **ptr_self) {
    NdArray *self = *ptr_self;
    NdArray *converted = NdArray_new(NULL, self->shape, DT_INT);

    double* cur_self = self->data;
    int* cur_conv = converted->data;

    for(int i = 0; i < converted->shape->len; i++) {
        cur_conv[i] = (int)cur_self[i];
    }

    NdArray_free(&self);
    *ptr_self = converted;
}

void NdArray_convert_type(NdArray **ptr_self, DataType datatype) {
    NdArray *self = *ptr_self;
    if(self->datatype == datatype) {
        return;
    }

    if(self->datatype == DT_INT && datatype == DT_DOUBLE) {
        _convert_datatype_from_int_to_double(ptr_self);
    } else if(self->datatype == DT_DOUBLE && datatype == DT_INT) {
        _convert_datatype_from_double_to_int(ptr_self);
    }
}

int _compare_int(int a, int b) {
    return a - b;
}

int _compare_double(double a, double b) {
    double temp = a - b;
    if(temp > 0) {
        return 1;
    } else if(temp < 0) {
        return -1;
    } else {
        return 0;
    }
}

int _compare_element(const void *a, const void *b, DataType datatype) {
    if(datatype == DT_INT) {
        return _compare_int(*(int*)a, *(int*)b);
    } else if(datatype == DT_DOUBLE) {
        return _compare_double(*(double*)a, *(double*)b);
    } else {
        abort();
    }
}

int _compare_element_scalar(const void *self, double value, DataType datatype) {
    if(datatype == DT_INT) {
        double temp = (double)(*(int*)self);
        return _compare_double(temp, value);
    } else if(datatype == DT_DOUBLE) {
        return _compare_double(*(double*)self, value);
    } else {
        abort();
    }
}

void _set_bool(void *cur, DataType datatype, int bool_value, CompareTag ct) {
    if(ct == CT_GT && bool_value <= 0) {
        memset(cur, 0, get_item_size(datatype));
    } else if(ct == CT_GE && bool_value < 0) {
        memset(cur, 0, get_item_size(datatype));
    } else if(ct == CT_LT && bool_value >= 0) {
        memset(cur, 0, get_item_size(datatype));
    } else if(ct == CT_LE && bool_value > 0) {
        memset(cur, 0, get_item_size(datatype));
    } else if(ct == CT_EQ && bool_value != 0) {
        memset(cur, 0, get_item_size(datatype));
    } else {
        *(char*)cur= 1;
    }
}

NdArray* NdArray_compare(NdArray *a, NdArray *b, CompareTag ct) {
    assert(a->shape->dim == b->shape->dim);
    assert(a->shape->len == b->shape->len);
    assert(a->datatype == b->datatype);

    NdArray *result = NdArray_new(NULL, a->shape, DT_BOOL);

    void *cur_a = a->data;
    void *cur_b = b->data;
    void *cur_result = result->data;

    int bool_value = 0;
    for(int i = 0; i < a->shape->len; i++) {
        int bool_value = _compare_element(cur_a, cur_b, a->datatype);
        _set_bool(cur_result, result->datatype, bool_value, ct);
        cur_a += a->item_size;
        cur_b += b->item_size;
        cur_result += result->item_size;
    }

    return result;
}

NdArray* NdArray_compare_scalar(NdArray *self, double value, CompareTag ct) {
    NdArray *result = NdArray_new(NULL, self->shape, DT_BOOL);

    void *cur_self = self->data;
    void *cur_result = result->data;

    int bool_value = 0;
    for(int i = 0; i < self->shape->len; i++) {
        int bool_value = _compare_element_scalar(cur_self, value, self->datatype);
        _set_bool(cur_result, result->datatype, bool_value, ct);
        cur_self += self->item_size;
        cur_result += result->item_size;
    }

    return result;
}

NdArray* NdArray_mask(NdArray *self, NdArray *mask) {
    NdArray* result = NdArray_new(NULL, self->shape, self->datatype);

    void* cur_self = self->data;
    void* cur_mask = mask->data;
    void* cur_result = result->data;

    for(int i = 0; i < self->shape->len; i++) {
        if(*(char*)cur_mask) {
            memcpy(cur_result, cur_self, self->item_size);
        }

        cur_self += self->item_size;
        cur_mask += mask->item_size;
        cur_result += result->item_size;
    }

    return result;
}
