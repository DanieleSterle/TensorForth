// Daniele Sterle SM3201594

// Tensor data structures and lifecycle.
// Handles memory allocation, reference counting, and shape operations like r, \, and #.

#include "tensor.h"
#include "utils.h"

int tensor_reshape(tensor* t, tensor* s) {

    if ((t  ==  NULL)  ||  (s  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    // PASARE [ n ] dovrebbe dare errore?
    int vector_result = is_vector(s);
    if (vector_result  !=  ERR_SUCCESS) return vector_result;

    int s_len = s->rows * s->columns;

    if ((s_len < 1)  ||  (s_len > 2)) {
        return ERR_SHAPE_MISMATCH; 
    }

    int new_rows;
    int new_cols;

    if (s_len == 2) {
        // shape is 2D Matrix
        new_rows = (int)s->values[0];
        new_cols = (int)s->values[1];
    } else {
        // shape is 1D Vector 
        new_rows = 1;
        new_cols = (int)s->values[0];
    }

    if ((new_rows <= 0)  ||  (new_cols <= 0)) {
        return ERR_SHAPE_MISMATCH;
    }

    int t_shape = t->rows * t->columns;
    int s_shape = new_rows * new_cols;
    
    if (t_shape  !=  s_shape) {
        return ERR_SHAPE_MISMATCH;
    }

    t->rows = new_rows;
    t->columns = new_cols;

    return ERR_SUCCESS;
}

int tensor_ravel(tensor* t) {

    if (t  ==  NULL) {
        return ERR_NULL_PTR;
    }

    int t_shape = t->rows * t->columns;

    t->rows = 1;
    t->columns = t_shape;

    return ERR_SUCCESS;
}

int tensor_get_shape(tensor* t, tensor* result) {

    if ((t  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    float* values = (float*) malloc(sizeof(float) * 2);

    if (values  ==  NULL) {
        return ERR_OUT_OF_MEMORY;
    }

    values[0] = t->rows;
    values[1] = t->columns;

    result->values = values;
    result->rows = 1;
    result->columns = 2;

    return ERR_SUCCESS;
}
