#include <stdio.h>
#include <stdlib.h>
#include "tester.h"
#include "ndarray.h"
#include "ndshape.h"

void test_ndarray_new() {
    NdShape *shape = NdShape_new(2, 2, 2);
    NdArray *array = NdArray_new(NULL, shape);
    NdArray_printArray(array);
}

void test_ndarray_new_with_data() {
    unsigned int data[2][2] = { {1, 2}, {3, 4} };
    NdShape *shape = NdShape_new(2, 2, 2);
    NdArray *array = NdArray_new(data, shape);
    NdArray_printArray(array);
}

void test_ndarray_copy() {
    unsigned int data[2][2] = { {1, 2}, {3, 4} };
    NdShape *shape = NdShape_new(2, 2, 2);
    NdArray *array = NdArray_new(data, shape);
    NdArray *copied = NdArray_copy(array);
    printf("array : ");
    NdArray_printArray(array);
    printf("copied : ");
    NdArray_printArray(copied);
}

void test_ndarray_aranges() {
    NdArray *array_zeros = NdArray_zeros(21);
    NdArray *array_ones = NdArray_ones(30);
    NdArray *array_arange = NdArray_arange(5, 50);

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

    NdArray *array0 = NdArray_new(data0, shape0);
    NdArray *array1 = NdArray_new(data1, shape1);
    NdArray *array2 = NdArray_arange(1, 121);

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
    NdArray *array = NdArray_arange(0, 729);
    NdArray_reshape(array, shape);

    unsigned int position[6] = {1, 1, 1, 1, 1, 0};
    int value = NdArray_getAt(array, position);
    printf("%d\n", value);

    int new_value = -1;
    NdArray_setAt(array, position, &new_value);
    NdArray_printArray(array);
}

void test_ndarray_matmul() {
    NdShape *shape = NdShape_new(2, 2, 2);
    NdArray *array = NdArray_arange(1,5);
    NdArray_reshape(array, shape);

    NdArray *result = NdArray_matmul(array, array);
    NdArray_printShape(result);
    NdArray_printArray(result);
    NdArray_free(&result);

    NdShape *shape0 = NdShape_new(5, 2, 3, 4, 5, 6);
    NdShape *shape1 = NdShape_new(5, 2, 3, 4, 6, 7);

    NdArray *array0 = NdArray_arange(1, 721);
    NdArray *array1 = NdArray_arange(1, 1009);

    NdArray_reshape(array0, shape0);
    NdArray_reshape(array1, shape1);
    
    result = NdArray_matmul(array0, array1);
    NdArray_printArray(result);
    NdArray_printShape(result);
}

int main() {
    //test("test_ndarray_new", test_ndarray_new);
    //test("test_ndarray_new_with_data", test_ndarray_new_with_data);
    //test("test_ndarray_copy", test_ndarray_copy);
    //test("test_ndarray_aranges", test_ndarray_aranges);
    //test("test_ndarray_reshape", test_ndarray_reshape);
    //test("test_ndarray_get_set", test_ndarray_get_set);
    test("test_ndarray_matmul", test_ndarray_matmul);
    return 0;
}
