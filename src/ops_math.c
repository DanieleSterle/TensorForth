// Daniele Sterle SM3201594

// Element-wise mathematics and logic.
// Contains OpenMP-parallelized loops for arithmetic, comparisons, and logic.

#include "ops_math.h"
#include "utils.h"

// ERROR CODE:
// 0 - sucess
// -1 - errore generico
// -2 - shape non conforme

int tensor_add(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    result->columns = t1->columns;
    result->rows = t1->rows;

    int s_values = result->rows * result->columns;
    result->values = malloc(sizeof(float) * s_values);

    if (result->values  ==  NULL) return ERR_OUT_OF_MEMORY; 

    /*
    OPTIMIZE WITH OPENMP
    for (int i = 0; i < res->rows; i++){
        for (int j = 0; j < res->columns; j++){
            res->values[i * res->columns + j] = t1->values[i * res->columns + j] + t2->values[i * res->columns + j];
        }
    }
    */

    for (int i = 0; i < s_values; i++) {
        result->values[i] = t1->values[i] + t2->values[i];
    }

    return ERR_SUCCESS;

}

int tensor_subtract(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    result->columns = t1->columns;
    result->rows = t1->rows;

    int s_values = result->rows * result->columns;
    result->values = malloc(sizeof(float) * s_values);

    if (result->values  ==  NULL) return ERR_OUT_OF_MEMORY; 

    /*
    OPTIMIZE WITH OPENMP
    for (int i = 0; i < res->rows; i++){
        for (int j = 0; j < res->columns; j++){
            res->values[i * res->columns + j] = t1->values[i * res->columns + j] - t2->values[i * res->columns + j];
        }
    }
    */

    for (int i = 0; i < s_values; i++) {
        result->values[i] = t1->values[i] - t2->values[i];
    }

    return ERR_SUCCESS;

}

int tensor_multiply(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    result->columns = t1->columns;
    result->rows = t1->rows;

    int s_values = result->rows * result->columns;
    result->values = malloc(sizeof(float) * s_values);

    if (result->values  ==  NULL) return ERR_OUT_OF_MEMORY; 

    /*
    OPTIMIZE WITH OPENMP
    for (int i = 0; i < res->rows; i++){
        for (int j = 0; j < res->columns; j++){
            res->values[i * res->columns + j] = t1->values[i * res->columns + j] * t2->values[i * res->columns + j];
        }
    }
    */

    for (int i = 0; i < s_values; i++) {
        result->values[i] = t1->values[i] * t2->values[i];
    }

    return ERR_SUCCESS;

}