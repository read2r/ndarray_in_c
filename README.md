
# N-Dimentional Array in C

Numpy의 다차원 배열(N-Dimentional Array)을 C언어로 모방한 개인 프로젝트입니다.

## 1. Array Attributes
### 1. 1 Struct memory layout
```
typedef struct _NdArray {
	DataType dataype;
	NdShape *shape;
	unsigned int item_size;
	unsigned int size;
	void *data;
} NdArray
```
* ndarray.datatype : Datatype including integer, double .
```
typedef enum _DataType {
	DT_INT,
	DT_DOUBLE
} Datatype
```
* ndarray.shape : Shape object of array including number of dimensions, array of dimension elements, total length of array elements. 
```
typedef struct _NdShape {
	unsigned int dim;
	unsigned int len;
	unsigned int *arr;
} NdShape
```
* item_size : Bytes comsumed by a element of the array.
* size : Total bytes comsumed by the elements of the array
* data : Array of The elements.

### 1.2 NdShape Functions
* NdShape_empty
* NdShape_new
* NdShape_set
* NdShape_copy
* NdShape_free
* NdShape_print
* NdShape_reshape
* NdShape_reverse

### 1.3 NdArray  Functions
* NdArray_new
* NdArray_copy
* NdArray_free
* NdArray_zeros
* NdArray_ones
* NdArray_arange
* NdArray_random
* NdArray_random_range
* NdArray_random_gaussian
* NdArray_chioce
* NdArray_reshape
* NdArray_reshape_fxied_array
* NdArray_reshape_variadic_args
* NdArray_get_at
* NdArray_set_at
* NdArray_subarray
* NdArray_print_array
* NdArray_print_shape
* NdArray_add
* NdArray_sub
* NdArray_mul
* NdArray_div
* NdArray_dot
* NdArray_matmul
* NdArray_compare
* NdArray_add_scalar
* NdArray_sub_scalar
* NdArray_mul_scalar
* NdArray_div_scalar
* NdArray_compare_scalar
* NdArray_mask
* NdArray_sum
* NdArray_mean
* NdArray_max
* NdArray_min
* NdArray_argmax
* NdArray_argmin
* NdArray_sum_axis
* NdArray_max_axis
* NdArray_min_axis
* NdArray_argmax_axis
* NdArray_argmin_axis
* NdArray_broadcast
* NdArray_convert_type
## 2. Numpy Documentary
https://numpy.org/doc/stable/reference/arrays.ndarray.html

