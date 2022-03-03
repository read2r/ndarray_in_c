#ifndef __ND_SHAPE_H__
#define __ND_SHAPE_H__

#include <stdarg.h>

typedef struct _NdShape {
    unsigned int dim;
    unsigned int len;
    unsigned int *arr;
} NdShape;

NdShape* NdShape_empty(unsigned int dim);
NdShape* NdShape_set(NdShape *self, unsigned int dim, ...);
NdShape* NdShape_set_fixed_array(NdShape *self, unsigned int dim, unsigned int *arr);
NdShape* NdShape_set_va_list(NdShape *self, unsigned int dim, va_list args);
//NdShape* NdShape_set_variadic_args(NdShape* self, unsigned int dim, ...);
NdShape* NdShape_new(unsigned int dim, ...);
NdShape* NdShape_new_fixed_array(unsigned int dim, unsigned int *arr);
//NdShape* NdShape_new_va_list(NdShape *self, unsigned int dim, va_list args);
//NdShape* NdShape_new_variadic_args(NdShape* self, unsigned int dim, ...);
NdShape* NdShape_copy(const NdShape *src);
void NdShape_free(NdShape **ptr_shape);
void NdShape_print(NdShape *self);
int NdShape_compare(NdShape *a, NdShape *b);
int NdShape_reshape(NdShape *dest, const NdShape *src);
int NdShape_reshape_fixed_array(NdShape *self, unsigned int dim, unsigned int *arr);
//int NdShape_reshape_va_lsit(NdShape *self, unsigned int dim, va_list args);
//int NdShape_reshape_variadic_args(NdShape *self, unsigned int dim, ...);
NdShape* NdShape_reverse(NdShape *self);

#endif
