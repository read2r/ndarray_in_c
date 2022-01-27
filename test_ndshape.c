#include <stdio.h>
#include <stdlib.h>
#include "tester.h"
#include "ndshape.h"

// print shape struct for the tests.
void NdShape_testPrintShape(const NdShape *ndshape) {
    printf("{ dim = %d, len = %d, shape = ", ndshape->dim, ndshape->len); 
    for(int i = 0; i < ndshape->dim; i++) {
        printf("%d", ndshape->arr[i]);
        if(i < ndshape->dim-1) {
            printf(", ");
        }
    }
    printf(" }\n");
}

void test_shape_empty() {
    NdShape *empty = NdShape_empty(1);
    NdShape_testPrintShape(empty);
}

void test_shape_set() {
    NdShape *shape = NdShape_empty(1);
    NdShape_testPrintShape(shape);
    NdShape_set(shape, 3, 1, 2, 3);
    NdShape_testPrintShape(shape);
    NdShape_set(shape, 6, 3, 3, 3, 3, 3, 3);
    NdShape_testPrintShape(shape);
}

void test_shape_new() {
    NdShape *shape = NdShape_new(3, 4, 4, 4);
    NdShape_testPrintShape(shape);
}

void test_shape_free() {
    NdShape *shape = NdShape_new(3, 4, 4, 4);
    unsigned int *shapeArray = shape->arr;
    NdShape_free(&shape);

    if(shape != NULL) {
        fprintf(stderr, "shape is not NULL\n");
    } else {
        printf("test ok\n");
    }
}

void test_shape_print() {
    NdShape *shape = NdShape_new(6, 2, 3, 4, 9, 8, 7);
    NdShape_testPrintShape(shape);
    NdShape_print(shape);
}

void test_shape_reshape() {
    NdShape *shape0 = NdShape_new(5, 2, 3, 4, 5, 6);
    NdShape *shape1 = NdShape_new(3, 6, 10, 12);
    
    printf("Before Reshape\n");
    NdShape_testPrintShape(shape0);
    NdShape_testPrintShape(shape1);

    printf("After Reshape\n");
    NdShape_reshape(shape0, shape1);
    NdShape_testPrintShape(shape0);
    NdShape_testPrintShape(shape1);
}

void test_shape_compare() {
    NdShape *shape0 = NdShape_new(3, 3, 3, 3);
    NdShape *shape1 = NdShape_new(3, 3, 3, 2);
    
    NdShape_testPrintShape(shape0);
    NdShape_testPrintShape(shape1);

    if(!NdShape_compare(shape0, shape1)) {
        fprintf(stderr, "both are not same\n");
    } else {
        printf("test ok\n");
    }
}

int main() {
    test("test_shape_empty", test_shape_empty);
    test("test_shape_set", test_shape_set);
    test("test_shape_new", test_shape_new);
    test("test_shape_free", test_shape_free);
    test("test_shape_print", test_shape_print);
    test("test_shape_reshape", test_shape_reshape);
    test("test_shape_compare", test_shape_compare);
    return 0;
}
