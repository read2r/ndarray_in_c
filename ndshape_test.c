#include <stdio.h>
#include <stdlib.h>
#include "ndshape.h"

// print shape struct for the tests.
void NdShape_testPrintShape(const NdShape *ndshape) {
    printf("{ dim = %d, len = %d, shape = ", ndshape->dim, ndshape->len); 
    for(int i = 0; i < ndshape->dim; i++) {
        printf("%d", ndshape->shape[i]);
        if(i < ndshape->dim-1) {
            printf(", ");
        }
    }
    printf(" }\n");
}

void testShapeEmpty() {
    printf("NdShape_empty test start\n");

    NdShape *empty = NdShape_empty(1);
    NdShape_testPrintShape(empty);

    printf("NdShape_empty test end\n\n");
}

void testShapeSet() {
    printf("NdShape_set test start\n");

    NdShape *shape = NdShape_empty(1);
    NdShape_testPrintShape(shape);
    NdShape_set(shape, 3, 1, 2, 3);
    NdShape_testPrintShape(shape);
    NdShape_set(shape, 6, 3, 3, 3, 3, 3, 3);
    NdShape_testPrintShape(shape);

    printf("NdShape_set test end\n\n");
}

void testShapeNew() {
    printf("NdShape_new test start\n");

    NdShape *shape = NdShape_new(3, 4, 4, 4);
    NdShape_testPrintShape(shape);

    printf("NdShape_new test end\n\n");
}

void testShapeFree() {
    printf("NdShape_free test start\n");

    NdShape *shape = NdShape_new(3, 4, 4, 4);
    unsigned int *shapeArray = shape->shape;
    NdShape_free(&shape);

    if(shape != NULL) {
        fprintf(stderr, "shape is not NULL\n");
    } else {
        printf("test ok\n");
    }

    printf("NdShape_free test end\n\n");
}

void testShapePrint() {
    printf("NdShape_print test start\n");

    NdShape *shape = NdShape_new(6, 2, 3, 4, 9, 8, 7);
    NdShape_testPrintShape(shape);
    NdShape_print(shape);

    printf("NdShape_print test end\n\n");
}

void testShapeReshape() {
    printf("NdShape_reshape test start\n");

    NdShape *shape0 = NdShape_new(5, 2, 3, 4, 5, 6);
    NdShape *shape1 = NdShape_new(3, 6, 10, 12);
    
    printf("Before Reshape\n");
    NdShape_testPrintShape(shape0);
    NdShape_testPrintShape(shape1);

    printf("After Reshape\n");
    NdShape_reshape(shape0, shape1);
    NdShape_testPrintShape(shape0);
    NdShape_testPrintShape(shape1);

    printf("NdShape_reshape test end\n\n");
}

void testShapeCompare() {
    printf("NdShape_compare test start\n");

    NdShape *shape0 = NdShape_new(3, 3, 3, 3);
    NdShape *shape1 = NdShape_new(3, 3, 3, 2);
    
    NdShape_testPrintShape(shape0);
    NdShape_testPrintShape(shape1);

    if(!NdShape_compare(shape0, shape1)) {
        fprintf(stderr, "both are not same\n");
    } else {
        printf("test ok\n");
    }

    printf("NdShape_compare test end\n\n");
}

int main() {
    testShapeEmpty();
    testShapeSet();
    testShapeNew();
    testShapeFree();
    testShapePrint();
    testShapeReshape();
    testShapeCompare();

    return 0;
}
