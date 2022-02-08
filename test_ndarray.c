#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tester.h"
#include "ndarray.h"
#include "ndshape.h"

void test_ndarray_new() {
    NdShape *shape = NdShape_new(2, 2, 2);
    NdArray *array = NdArray_new(NULL, shape, DT_INT);
    NdArray_printArray(array);
}

void test_ndarray_new_with_data() {
    unsigned int data[2][2] = { {1, 2}, {3, 4} };
    NdShape *shape = NdShape_new(2, 2, 2);
    NdArray *array = NdArray_new(data, shape, DT_INT);
    NdArray_printArray(array);
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
    printf("after reshaping array0 : ");
    NdArray_reshape(array0, shape2);
    NdArray_printShape(array0);
    NdArray_printArray(array0);
    printf("\n");

    printf("before reshaping array1 : ");
    NdArray_printShape(array1);
    NdArray_printArray(array1);
    NdArray_reshape(array1, shape3);
    printf("after reshaping array1 : ");
    NdArray_printShape(array1);
    NdArray_printArray(array1);
    printf("\n");

    printf("before reshaping array2 : ");
    NdArray_printShape(array2);
    NdArray_printArray(array2);
    printf("after reshaping array1 to 2 dimesions array : ");
    NdArray_reshape(array2, shape4);
    NdArray_printShape(array2);
    NdArray_printArray(array2);
    printf("after reshaping array1 to 3 dimesions array : ");
    NdArray_reshape(array2, shape5);
    NdArray_printShape(array2);
    NdArray_printArray(array2);
    printf("after reshaping array1 to 5 dimesions array : ");
    NdArray_reshape(array2, shape6);
    NdArray_printShape(array2);
    NdArray_printArray(array2);
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
}

void test_ndarray_matmul() {
    NdShape *shape, *shape0, *shape1, *shape_result;
    NdArray *array, *array0, *array1, *array_result;

    // simple matmul test.
    shape = NdShape_new(2, 2, 2);
    array = NdArray_arange(1, 5, DT_INT);
    NdArray_reshape(array, shape);

    array_result = NdArray_matmul(array, array);
    NdArray_printShape(array_result);
    NdArray_printArray(array_result);

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

    NdArray_free(&array0);
    NdArray_free(&array1);
    NdArray_free(&array_result);
}

void test_ndarray_dot() {
    NdShape *shape, *shape0, *shape1, *shape_result;
    NdArray *array, *array0, *array1, *array_result;

    // simple dot produect test.
    shape = NdShape_new(2, 2, 2);
    array = NdArray_arange(1, 5, DT_INT);
    NdArray_reshape(array, shape);

    array_result = NdArray_dot(array, array);
    NdArray_printShape(array_result);
    NdArray_printArray(array_result);

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

    NdArray_free(&array0);
    NdArray_free(&array1);
    NdArray_free(&array_result);
}

void test_ndarray_matmul_float() {
    NdShape *shape0, *shape1, *shape2, *shape3, *shape_result;
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
}

void test_ndarray_dot_float() {
    NdShape *shape0, *shape1, *shape2, *shape3, *shape_result;
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
    return 0;
}
