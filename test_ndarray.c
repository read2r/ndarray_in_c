#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "tester.h"
#include "ndarray.h"
#include "ndshape.h"

void test_ndarray_new() {
    NdShape *shape = NdShape_new(2, 2, 2);
    NdArray *array = NdArray_new(NULL, shape, DT_INT);

    NdArray_printArray(array);

    NdShape_free(&shape);
    NdArray_free(&array);
}

void test_ndarray_new_with_data() {
    unsigned int data[2][2] = { {1, 2}, {3, 4} };
    NdShape *shape = NdShape_new(2, 2, 2);
    NdArray *array = NdArray_new(data, shape, DT_INT);

    NdArray_printArray(array);

    NdShape_free(&shape);
    NdArray_free(&array);
}

void test_ndarray_copy() {
    unsigned int data[2][2] = { {1, 2}, {3, 4} };
    NdShape *shape = NdShape_new(2, 2, 2);
    NdArray *array = NdArray_new(data, shape, DT_INT);
    NdArray *copied = NdArray_copy(array);

    printf("array : ");
    NdArray_printArray(array);
    printf("copied : ");
    NdArray_printArray(copied);

    NdShape_free(&shape);
    NdArray_free(&array);
    NdArray_free(&copied);
}

void test_ndarray_aranges() {
    NdArray *array_zeros = NdArray_zeros(21, DT_INT);
    NdArray *array_ones = NdArray_ones(30, DT_INT);
    NdArray *array_arange = NdArray_arange(5, 50, DT_INT);

    printf("zeors  : ");
    NdArray_printArray(array_zeros);
    printf("ones   : ");
    NdArray_printArray(array_ones);
    printf("arange : ");
    NdArray_printArray(array_arange);

    NdArray_free(&array_zeros);
    NdArray_free(&array_ones);
    NdArray_free(&array_arange);
}

void test_ndarray_reshape() {
    int data0[3][3][2] = { 
        { {70, 80}, {94, 80}, {70, 85} },
        { {83, 90}, {95, 60}, {90, 82} },
        { {98, 89}, {99, 94}, {91, 87} }
    };
    
    int data1[2][3][4][3] = {
        { { { 1111, 1112, 1113 }, { 1121, 1122, 1123 }, { 1131, 1132, 1133 }, { 1141, 1142, 1143 }, },
            { { 1211, 1212, 1213 }, { 1221, 1222, 1223 }, { 1231, 1232, 1233 }, { 1241, 1242, 1243 }, },
            { { 1311, 1312, 1313 }, { 1321, 1322, 1323 }, { 1331, 1332, 1333 }, { 1341, 1342, 1343 }, }
        },
        {
            { { 2111, 2112, 2113 }, { 2121, 2122, 2123 }, { 2131, 2132, 2133 }, { 2141, 2142, 2143 }, },
            { { 2211, 2212, 2213 }, { 2221, 2222, 2223 }, { 2231, 2232, 2233 }, { 2241, 2242, 2243 }, },
            { { 2311, 2312, 2313 }, { 2321, 2322, 2323 }, { 2331, 2332, 2333 }, { 2341, 2342, 2343 }, }
        }
    };

    NdShape *shape0 = NdShape_new(3, 3, 3, 2);
    NdShape *shape1 = NdShape_new(4, 2, 3, 4, 3);
    NdShape *shape2 = NdShape_new(2, 3, 6);
    NdShape *shape3 = NdShape_new(2, 8, 9);
    NdShape *shape4 = NdShape_new(2, 12, 10);
    NdShape *shape5 = NdShape_new(3, 4, 6, 5);
    NdShape *shape6 = NdShape_new(5, 2, 2, 2, 3, 5);

    NdArray *array0 = NdArray_new(data0, shape0, DT_INT);
    NdArray *array1 = NdArray_new(data1, shape1, DT_INT);
    NdArray *array2 = NdArray_arange(1, 121, DT_INT);

    printf("before reshaping array0 : ");
    NdArray_printShape(array0);
    NdArray_printArray(array0);
    printf("\n");
    printf("after reshaping array0 : ");
    NdArray_reshape(array0, shape2);
    NdArray_printShape(array0);
    NdArray_printArray(array0);
    printf("\n\n");

    printf("before reshaping array1 : ");
    NdArray_printShape(array1);
    NdArray_printArray(array1);
    NdArray_reshape(array1, shape3);
    printf("\n");
    printf("after reshaping array1 : ");
    NdArray_printShape(array1);
    NdArray_printArray(array1);
    printf("\n\n");

    printf("before reshaping array2 : ");
    NdArray_printShape(array2);
    NdArray_printArray(array2);
    printf("\n");
    printf("after reshaping array1 to 2 dimesions array : ");
    NdArray_reshape(array2, shape4);
    NdArray_printShape(array2);
    NdArray_printArray(array2);
    printf("\n");
    printf("after reshaping array1 to 3 dimesions array : ");
    NdArray_reshape(array2, shape5);
    NdArray_printShape(array2);
    NdArray_printArray(array2);
    printf("\n");
    printf("after reshaping array1 to 5 dimesions array : ");
    NdArray_reshape(array2, shape6);
    NdArray_printShape(array2);
    NdArray_printArray(array2);

    NdShape_free(&shape0);
    NdShape_free(&shape1);
    NdShape_free(&shape2);
    NdShape_free(&shape3);
    NdShape_free(&shape4);
    NdShape_free(&shape5);
    NdShape_free(&shape6);

    NdArray_free(&array0);
    NdArray_free(&array1);
    NdArray_free(&array2);
}

void test_ndarray_get_set() {
    NdShape *shape = NdShape_new(6, 3, 3, 3, 3, 3, 3);
    NdArray *array = NdArray_arange(0, 729, DT_INT);
    NdArray_reshape(array, shape);

    unsigned int position[6] = {1, 1, 1, 1, 1, 0};
    int* ptr_value = (int*)NdArray_getAt(array, position);
    printf("%d\n", *ptr_value);

    int new_value = -1;
    NdArray_setAt(array, position, &new_value);
    NdArray_printArray(array);

    NdShape_free(&shape);
    NdArray_free(&array);
}

void test_ndarray_matmul() {
    NdShape *shape, *shape0, *shape1;
    NdArray *array, *array0, *array1, *array_result;

    // simple matmul test.
    shape = NdShape_new(2, 2, 2);
    array = NdArray_arange(1, 5, DT_INT);
    NdArray_reshape(array, shape);

    array_result = NdArray_matmul(array, array);
    NdArray_printShape(array_result);
    NdArray_printArray(array_result);

    NdShape_free(&shape);
    NdArray_free(&array);
    NdArray_free(&array_result);

    // two arrays matmul test, taht have same N dimensions.
    shape0 = NdShape_new(5, 2, 3, 4, 5, 6);
    shape1 = NdShape_new(5, 2, 3, 4, 6, 7);

    array0 = NdArray_arange(1, 721, DT_INT);
    array1 = NdArray_arange(1, 1009, DT_INT);

    NdArray_reshape(array0, shape0);
    NdArray_reshape(array1, shape1);
    
    array_result = NdArray_matmul(array0, array1);
    NdArray_printArray(array_result);
    NdArray_printShape(array_result);

    NdShape_free(&shape0);
    NdShape_free(&shape1);
    NdArray_free(&array0);
    NdArray_free(&array1);
    NdArray_free(&array_result);

    // two arrays matmul test(1), that have different N dimesions.
    shape0 = NdShape_new(5, 3, 3, 3, 3, 10);
    shape1 = NdShape_new(3, 3, 10, 5);

    array0 = NdArray_arange(1, 811, DT_INT);
    array1 = NdArray_arange(1, 151, DT_INT);

    NdArray_reshape(array0, shape0);
    NdArray_reshape(array1, shape1);
    
    array_result = NdArray_matmul(array0, array1);
    NdArray_printArray(array_result);
    NdArray_printShape(array_result);

    NdShape_free(&shape0);
    NdShape_free(&shape1);
    NdArray_free(&array0);
    NdArray_free(&array1);
    NdArray_free(&array_result);

    // two arrays matmul test(2), that have different N dimesions.
    shape0 = NdShape_new(3, 3, 3, 10);
    shape1 = NdShape_new(6, 2, 2, 3, 3, 10, 2);

    array0 = NdArray_arange(1, 91, DT_INT);
    array1 = NdArray_arange(1, 721, DT_INT);

    NdArray_reshape(array0, shape0);
    NdArray_reshape(array1, shape1);
    
    array_result = NdArray_matmul(array0, array1);
    NdArray_printArray(array_result);
    NdArray_printShape(array_result);

    NdShape_free(&shape0);
    NdShape_free(&shape1);
    NdArray_free(&array0);
    NdArray_free(&array1);
    NdArray_free(&array_result);
}

void test_ndarray_dot() {
    NdShape *shape, *shape0, *shape1;
    NdArray *array, *array0, *array1, *array_result;

    // simple dot produect test.
    shape = NdShape_new(2, 2, 2);
    array = NdArray_arange(1, 5, DT_INT);
    NdArray_reshape(array, shape);

    array_result = NdArray_dot(array, array);
    NdArray_printShape(array_result);
    NdArray_printArray(array_result);

    NdShape_free(&shape);
    NdArray_free(&array);
    NdArray_free(&array_result);
    
    // two arrays dot product test, that have same N dimesions.
    shape0 = NdShape_new(4, 3, 3, 3, 6);
    shape1 = NdShape_new(4, 4, 4, 6, 4);

    array0 = NdArray_arange(1, 163, DT_INT);
    array1 = NdArray_arange(1, 385, DT_INT);

    NdArray_reshape(array0, shape0);
    NdArray_reshape(array1, shape1);
    
    array_result = NdArray_dot(array0, array1);
    NdArray_printArray(array_result);
    NdArray_printShape(array_result);

    NdShape_free(&shape0);
    NdShape_free(&shape1);
    NdArray_free(&array0);
    NdArray_free(&array1);
    NdArray_free(&array_result);

    // two arrays dot product test(1), that have different N dimesions.
    shape0 = NdShape_new(5, 3, 3, 3, 3, 10);
    shape1 = NdShape_new(3, 2, 10, 2);

    array0 = NdArray_arange(1, 811, DT_INT);
    array1 = NdArray_arange(1, 41, DT_INT);

    NdArray_reshape(array0, shape0);
    NdArray_reshape(array1, shape1);
    
    array_result = NdArray_dot(array0, array1);
    NdArray_printArray(array_result);
    NdArray_printShape(array_result);

    NdShape_free(&shape0);
    NdShape_free(&shape1);
    NdArray_free(&array0);
    NdArray_free(&array1);
    NdArray_free(&array_result);

    // two arrays dot product test(2), that have different N dimesions.
    shape0 = NdShape_new(3, 3, 3, 10);
    shape1 = NdShape_new(6, 2, 2, 2, 2, 10, 2);

    array0 = NdArray_arange(1, 91, DT_INT);
    array1 = NdArray_arange(1, 321, DT_INT);

    NdArray_reshape(array0, shape0);
    NdArray_reshape(array1, shape1);
    
    array_result = NdArray_dot(array0, array1);
    NdArray_printArray(array_result);
    NdArray_printShape(array_result);

    NdShape_free(&shape0);
    NdShape_free(&shape1);
    NdArray_free(&array0);
    NdArray_free(&array1);
    NdArray_free(&array_result);
}

void test_ndarray_matmul_float() {
    NdShape *shape0, *shape1, *shape2, *shape3;
    NdArray *array0, *array1, *array2, *array3, *array_result;

    double data0[2][2] = { {1.0, 1.1}, {2.0, 2.1} };
    double data1[2][4] = { {1.0, 1.1, 1.2, 1.3}, {2.0, 2.1, 2.2, 2.3} };
    
    shape0 = NdShape_new(2, 2, 2);
    shape1 = NdShape_new(2, 2, 4);

    array0 = NdArray_new(data0, shape0, DT_DOUBLE);
    array1 = NdArray_new(data1, shape1, DT_DOUBLE);

    array_result = NdArray_dot(array0, array1);
    NdArray_printArray(array_result);

    shape2 = NdShape_new(5, 2, 3, 4, 5, 6);
    shape3 = NdShape_new(5, 2, 3, 4, 6, 7);

    array2 = NdArray_arange(1, 721, DT_DOUBLE);
    array3 = NdArray_arange(1, 1009, DT_DOUBLE);

    NdArray_add_scalar(array2, 0.1234);
    NdArray_sub_scalar(array3, 0.4321);

    NdArray_reshape(array2, shape2);
    NdArray_reshape(array3, shape3);
    
    array_result = NdArray_matmul(array2, array3);
    NdArray_printArray(array_result);
    NdArray_printShape(array_result);

    NdShape_free(&shape0);
    NdShape_free(&shape1);
    NdShape_free(&shape2);
    NdShape_free(&shape3);
    NdArray_free(&array0);
    NdArray_free(&array1);
    NdArray_free(&array2);
    NdArray_free(&array3);
    NdArray_free(&array_result);
}

void test_ndarray_dot_float() {
    NdShape *shape0, *shape1, *shape2, *shape3;
    NdArray *array0, *array1, *array2, *array3, *array_result;

    double data0[2][2] = { {1.0, 1.1}, {2.0, 2.1} };
    double data1[2][4] = { {1.0, 1.1, 1.2, 1.3}, {2.0, 2.1, 2.2, 2.3} };
    
    shape0 = NdShape_new(2, 2, 2);
    shape1 = NdShape_new(2, 2, 4);

    array0 = NdArray_new(data0, shape0, DT_DOUBLE);
    array1 = NdArray_new(data1, shape1, DT_DOUBLE);

    array_result = NdArray_dot(array0, array1);
    NdArray_printArray(array_result);

    shape2 = NdShape_new(4, 3, 3, 3, 6);
    shape3 = NdShape_new(4, 4, 4, 6, 4);

    array2 = NdArray_arange(1, 163, DT_DOUBLE);
    array3 = NdArray_arange(1, 385, DT_DOUBLE);

    NdArray_add_scalar(array2, 0.1234);
    NdArray_sub_scalar(array3, 0.4321);

    NdArray_reshape(array2, shape2);
    NdArray_reshape(array3, shape3);
        
    array_result = NdArray_dot(array2, array3);
    NdArray_printArray(array_result);
    NdArray_printShape(array_result);

    NdShape_free(&shape0);
    NdShape_free(&shape1);
    NdShape_free(&shape2);
    NdShape_free(&shape3);
    NdArray_free(&array0);
    NdArray_free(&array1);
    NdArray_free(&array2);
    NdArray_free(&array3);
    NdArray_free(&array_result);
}

void* sigmoid(void *ptr_x) {
    double x = *(double*)ptr_x;
    *(double*)ptr_x = 1 + 1 / exp(-x);
    return ptr_x;
}

void test_ndarray_operations() {
    NdArray *array = NdArray_arange(1, 10, DT_DOUBLE);
    double a = 0.1;

    NdArray_add_scalar(array, a);
    NdArray_printArray(array);
    NdArray_sub_scalar(array, a);
    NdArray_printArray(array);
    NdArray_mul_scalar(array, a);
    NdArray_printArray(array);
    NdArray_div_scalar(array, a);
    NdArray_printArray(array);
    printf("\n");

    NdArray *array0 = NdArray_copy(array);
    NdArray *array1 = NdArray_copy(array);

    NdArray_add(array0, array1);
    NdArray_printArray(array0);
    NdArray_sub(array0, array1);
    NdArray_printArray(array0);
    NdArray_mul(array0, array1);
    NdArray_printArray(array0);
    NdArray_div(array0, array1);
    NdArray_printArray(array0);
    printf("\n");

    NdArray_broadcast(array, sigmoid);
    NdArray_printArray(array);

    NdShape *shape2 = NdShape_new(4, 3, 4, 5, 6);
    NdShape *shape3 = NdShape_new(2, 5, 6);

    NdArray *array2 = NdArray_ones(3 * 4 * 5 * 6, DT_DOUBLE);
    NdArray *array3 = NdArray_arange(1, 5 * 6 + 1, DT_DOUBLE);

    NdArray_reshape(array2, shape2);
    NdArray_reshape(array3, shape3);

    NdArray_mul_scalar(array3, 2);

    NdArray_printArray(array1);
    NdArray_printArray(array2);

    NdArray_add(array2, array3); 
    printf("add : ");
    NdArray_printArray(array2);
    printf("\n");

    NdArray_sub(array2, array3); 
    printf("sub : ");
    NdArray_printArray(array2);
    printf("\n");

    NdArray_mul(array2, array3); 
    printf("mul : ");
    NdArray_printArray(array2);
    printf("\n");

    NdArray_div(array2, array3); 
    printf("div : ");
    NdArray_printArray(array2);
    printf("\n");

    NdShape_free(&shape2);
    NdShape_free(&shape3);

    NdArray_free(&array);
    NdArray_free(&array0);
    NdArray_free(&array1);
    NdArray_free(&array2);
    NdArray_free(&array3);
}

void test_ndarray_subarray() {
    NdShape *shape = NdShape_new(4, 10, 2, 3, 4);
    NdArray *array = NdArray_arange(0, 10 * 2 * 3 * 4, DT_INT);
    NdArray_reshape(array, shape);
    NdArray_printArray(array);
    printf("\n");

    NdArray *indices = NdArray_choice(3, 10, DT_INT);
    for(int i = 0; i < 3; i++) {
        int *idx = ((int*)indices->data + i);
        printf("[%d] >> ", *idx);
        NdArray *subarray = NdArray_subarray(array, (unsigned int *)idx, 1);
        NdArray_printShape(subarray);
        NdArray_printArray(subarray);
        NdArray_sub_free(&subarray);
    }
    printf("\n");

    unsigned int position[3] = { 5, 1 };
    NdArray *subarray = NdArray_subarray(array, position, 2);
    NdArray_printShape(subarray);
    NdArray_printArray(subarray);
    NdArray_sub_free(&subarray);

    NdShape_free(&shape);
    NdArray_free(&array);
    NdArray_free(&indices);
}

void test_ndarray_operations2() {
    int *ptr_value;
    int idx;
    NdArray *array = NdArray_arange(1, 11, DT_INT);

    ptr_value = NdArray_sum(array);
    printf("%d\n", *ptr_value);
    free(ptr_value);

    ptr_value = NdArray_max(array);
    idx = NdArray_argmax(array);
    printf("[%d] %d\n", idx, *ptr_value);
    free(ptr_value);

    ptr_value = NdArray_min(array);
    idx = NdArray_argmin(array);
    printf("[%d] %d\n", idx, *ptr_value);
    free(ptr_value);

    ptr_value = NdArray_mean(array);
    printf("%d\n", *ptr_value);
    free(ptr_value);
    printf("\n");

    NdArray_free(&array);
}

void test_ndarray_random() {
    NdArray *array_int = NdArray_random(10, DT_INT);
    NdArray *array_double = NdArray_random(10, DT_DOUBLE);
    NdArray_printArray(array_int);
    NdArray_printArray(array_double);
    NdArray_free(&array_int);
    NdArray_free(&array_double);

    sleep(1);

    NdArray *array_int_range = NdArray_random_range(20, 0, 10, DT_INT);
    NdArray *array_double_range = NdArray_random_range(20, 0, 10, DT_DOUBLE);
    NdArray_printArray(array_int_range);
    NdArray_printArray(array_double_range);
    NdArray_free(&array_int_range);
    NdArray_free(&array_double_range);

    array_int_range = NdArray_random_range(10, 30, 50, DT_INT);
    array_double_range = NdArray_random_range(10, 30, 50, DT_DOUBLE);
    NdArray_printArray(array_int_range);
    NdArray_printArray(array_double_range);
    NdArray_free(&array_int_range);
    NdArray_free(&array_double_range);

    NdShape *shape_random = NdShape_new(2, 100, 784);
    NdArray *array_random = NdArray_random(100 * 784, DT_DOUBLE);
    NdArray_reshape(array_random, shape_random);
    NdArray_printArray(array_random);
    NdArray_printShape(array_random);
    NdShape_free(&shape_random);
    NdArray_free(&array_random);
}

void test_ndarray_choice() {
    NdArray *choice0 = NdArray_choice(100, 100, DT_INT);
    NdArray_printArray(choice0);
    NdArray_free(&choice0);

    NdArray *choice1 = NdArray_choice(100, 60000, DT_INT);
    NdArray_printArray(choice1);
    NdArray_free(&choice1);
}

void test_ndarray_transpose() {
    NdShape *shape = NdShape_new(4, 2, 3, 4, 5);
    NdArray *array = NdArray_arange(0, 2 * 3 * 4 * 5, DT_INT);
    NdArray_reshape(array, shape);

    NdArray *transposed = NdArray_transpose(array);

    NdArray_printArray(array);
    NdArray_printArray(transposed);

    NdShape_free(&shape);
    NdArray_free(&array);
    NdArray_free(&transposed);
}

void test_ndarray_sum_axis() {
    NdShape *shape = NdShape_new(5, 2, 4, 7, 6, 5);
    NdArray *array = NdArray_arange(0, 2 * 4 * 7 * 6 * 5, DT_INT);
    NdArray_reshape(array, shape);
    
    NdArray *result;
    result = NdArray_sum_axis(array, 0);
    NdArray_printArray(result);
    NdArray_printShape(result);
    NdArray_free(&result);
    printf("\n");

    result = NdArray_sum_axis(array, 1);
    NdArray_printArray(result);
    NdArray_printShape(result);
    NdArray_free(&result);
    printf("\n");

    result = NdArray_sum_axis(array, 2);
    NdArray_printArray(result);
    NdArray_printShape(result);
    NdArray_free(&result);
    printf("\n");

    result = NdArray_sum_axis(array, 3);
    NdArray_printArray(result);
    NdArray_printShape(result);
    NdArray_free(&result);
    printf("\n");

    result = NdArray_sum_axis(array, 4);
    NdArray_printArray(result);
    NdArray_printShape(result);
    NdArray_free(&result);
    printf("\n");

    NdShape_free(&shape);
    NdArray_free(&array);
}

void test_ndarray_convert_datatype() {
    NdArray *array = NdArray_arange(0, 10, DT_INT);

    NdArray_convert_type(&array, DT_INT);
    NdArray_printArray(array);

    NdArray_convert_type(&array, DT_DOUBLE);
    NdArray_printArray(array);

    NdArray_add_scalar(array, 0.1234);
    NdArray_printArray(array);

    NdArray_convert_type(&array, DT_INT);
    NdArray_printArray(array);

    NdArray_free(&array);
}

void test_ndarray_compare() {
    NdArray *a = NdArray_arange(0, 20, DT_INT);
    NdArray *b = NdArray_ones(20, DT_INT);
    NdArray_mul_scalar(b, 10);

    NdArray_printArray(a);
    NdArray_printArray(b);
    printf("\n");

    NdArray *mask;
    NdArray *result;

    mask = NdArray_compare(a, b, CT_GT);
    result = NdArray_mask(a, mask);
    NdArray_printArray(result);
    NdArray_free(&result);
    NdArray_free(&mask);

    mask = NdArray_compare(a, b, CT_GE);
    result = NdArray_mask(a, mask);
    NdArray_printArray(result);
    NdArray_free(&result);
    NdArray_free(&mask);

    mask = NdArray_compare(a, b, CT_EQ);
    result = NdArray_mask(a, mask);
    NdArray_printArray(result);
    NdArray_free(&result);
    NdArray_free(&mask);

    mask = NdArray_compare(a, b, CT_LE);
    result = NdArray_mask(a, mask);
    NdArray_printArray(result);
    NdArray_free(&result);
    NdArray_free(&mask);

    mask = NdArray_compare(a, b, CT_LT);
    result = NdArray_mask(a, mask);
    NdArray_printArray(result);
    NdArray_free(&result);
    NdArray_free(&mask);

    NdArray_free(&a);
    NdArray_free(&b);
}

void test_ndarray_compare_scalar() {
    NdArray *a = NdArray_arange(0, 20, DT_INT);
    double s = 10;

    NdArray_printArray(a);
    printf("\n");

    NdArray *mask;
    NdArray *result;

    mask = NdArray_compare_scalar(a, s, CT_GT);
    result = NdArray_mask(a, mask);
    NdArray_printArray(result);
    NdArray_free(&result);
    NdArray_free(&mask);

    mask = NdArray_compare_scalar(a, s, CT_GE);
    result = NdArray_mask(a, mask);
    NdArray_printArray(result);
    NdArray_free(&result);
    NdArray_free(&mask);

    mask = NdArray_compare_scalar(a, s, CT_EQ);
    result = NdArray_mask(a, mask);
    NdArray_printArray(result);
    NdArray_free(&result);
    NdArray_free(&mask);

    mask = NdArray_compare_scalar(a, s, CT_LE);
    result = NdArray_mask(a, mask);
    NdArray_printArray(result);
    NdArray_free(&result);
    NdArray_free(&mask);

    mask = NdArray_compare_scalar(a, s, CT_LT);
    result = NdArray_mask(a, mask);
    NdArray_printArray(result);
    NdArray_free(&result);
    NdArray_free(&mask);

    NdArray_free(&a);
}

void test_ndarray_random_gaussian() {
    NdArray *array = NdArray_random_gaussian(10);
    NdArray_printArray(array);
    NdArray_free(&array);
}

void test_ndarray_argmax_axis() {
    NdShape *shape = NdShape_new(4, 2, 3, 4, 5);
    NdArray *array = NdArray_arange(0, 2 * 3 * 4 * 5, DT_DOUBLE);
    NdArray_reshape(array, shape);

    //NdArray_printArray(array);
    NdArray_printShape(array);
    printf("\n");
    
    NdArray *result = NdArray_argmax_axis(array, 0);
    NdArray_printArray(result);
    NdArray_printShape(result);
    NdArray_free(&result);
    printf("\n");

    result = NdArray_argmax_axis(array, 1);
    NdArray_printArray(result);
    NdArray_printShape(result);
    NdArray_free(&result);
    printf("\n");

    result = NdArray_argmax_axis(array, 2);
    NdArray_printArray(result);
    NdArray_printShape(result);
    NdArray_free(&result);
    printf("\n");

    result = NdArray_argmax_axis(array, 3);
    NdArray_printArray(result);
    NdArray_printShape(result);
    NdArray_free(&result);
    printf("\n");

    NdShape_free(&shape);
    NdArray_free(&array);
}

void test_ndarray_max_axis() {
    NdShape *shape = NdShape_new(4, 2, 3, 4, 5);
    NdArray *array = NdArray_arange(0, 2 * 3 * 4 * 5, DT_DOUBLE);
    NdArray_reshape(array, shape);

    //NdArray_printArray(array);
    NdArray_printShape(array);
    printf("\n");
    
    NdArray *result = NdArray_max_axis(array, 0);
    NdArray_printArray(result);
    NdArray_printShape(result);
    NdArray_free(&result);
    printf("\n");

    result = NdArray_max_axis(array, 1);
    NdArray_printArray(result);
    NdArray_printShape(result);
    NdArray_free(&result);
    printf("\n");

    result = NdArray_max_axis(array, 2);
    NdArray_printArray(result);
    NdArray_printShape(result);
    NdArray_free(&result);
    printf("\n");

    result = NdArray_max_axis(array, 3);
    NdArray_printArray(result);
    NdArray_printShape(result);
    NdArray_free(&result);
    printf("\n");

    NdShape_free(&shape);
    NdArray_free(&array);
}

void test_ndarray_reshape_variadic() {
    NdArray *array = NdArray_arange(0, 12, DT_INT);
    NdArray_printArray(array);
    NdArray_printShape(array);

    NdArray_reshape_variadic(array, 2, 3, 4);
    NdArray_printArray(array);
    NdArray_printShape(array);

    NdArray_free(&array);
}

int main() {
    test("test_ndarray_new", test_ndarray_new);
    test("test_ndarray_new_with_data", test_ndarray_new_with_data);
    test("test_ndarray_copy", test_ndarray_copy);
    test("test_ndarray_aranges", test_ndarray_aranges);
    test("test_ndarray_reshape", test_ndarray_reshape);
    test("test_ndarray_get_set", test_ndarray_get_set);
    test("test_ndarray_matmul", test_ndarray_matmul);
    test("test_ndarray_dot", test_ndarray_dot);
    test("test_ndarray_matul_float", test_ndarray_matmul_float);
    test("test_ndarray_dot_float", test_ndarray_dot_float);
    test("test_ndarray_operations", test_ndarray_operations);
    test("test_ndarray_subarray", test_ndarray_subarray);
    test("test_ndarray_operations2 ", test_ndarray_operations2);
    test("test_ndarray_random", test_ndarray_random);
    test("test_ndarray_choice", test_ndarray_choice);
    test("test_ndarray_transpose", test_ndarray_transpose);
    test("test_ndarray_sum_axis", test_ndarray_sum_axis);
    test("test_ndarray_convert_datatype", test_ndarray_convert_datatype);
    test("test_ndarray_compare", test_ndarray_compare);
    test("test_ndarray_compare_scalar", test_ndarray_compare_scalar);
    test("test_ndarray_random_gaussian", test_ndarray_random_gaussian);
    test("test_ndarray_argmax_axis", test_ndarray_argmax_axis);
    test("test_ndarray_max_axis", test_ndarray_max_axis);
    test("test_ndarray_reshape_variadic", test_ndarray_reshape_variadic);
    return 0;
}
