#ifndef __ND_SHAPE_H__
#define __ND_SHAPE_H__

typedef struct tagNdShape {
    unsigned int dim;
    unsigned int len;
    unsigned int *arr;
} NdShape;

NdShape* NdShape_empty(unsigned int dim);
NdShape* NdShape_set(NdShape *ndarray, unsigned int dim, ...);
NdShape* NdShape_new(unsigned int dim, ...);
NdShape* NdShape_copy(const NdShape *src);
void NdShape_free(NdShape **ptr_ndshape);
void NdShape_print(NdShape *ndshape);
int NdShape_compareDimension(NdShape *a, NdShape *b);
int NdShape_compareLength(NdShape *a, NdShape *b);
int NdShape_compareShapeArray(NdShape *a, NdShape *b);
int NdShape_compare(NdShape *a, NdShape *b);
int NdShape_reshape(NdShape *dest, const NdShape *src);
NdShape* NdShape_reverse(NdShape *self);
//int NdShape_swapaxes();
//unsigned int* NdShape_getShapeArray(const NdShape *ndshape);
//unsigned int NdShape_getDimension(const NdShape *ndshape);
//unsigned int NdShape_getLength(const NdShape *ndshape);

#endif
