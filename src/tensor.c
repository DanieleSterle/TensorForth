// Daniele Sterle SM3201594

// Tensor data structures and lifecycle.
// Handles memory allocation, reference counting, and shape operations like r, \, and #.

#include "tensor.h"
#include "utils.h"

int tensor_reshape(tensor* t, tensor* s) {

    if ((t  ==  NULL)  ||  (s  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t);
    ASSERT_NUMERIC(s);

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

    ASSERT_NUMERIC(t);

    int t_shape = t->rows * t->columns;

    t->rows = 1;
    t->columns = t_shape;

    return ERR_SUCCESS;
}

int tensor_get_shape(tensor* t, tensor* result) {

    if ((t  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t);

    int create_numeric_tensor_result = create_numeric_tensor(result, NULL, 1, 2);
    if (create_numeric_tensor_result  !=  ERR_SUCCESS) {
        return create_numeric_tensor_result;
    }

    result->values[0] = t->rows;
    result->values[1] = t->columns;

    return ERR_SUCCESS;
}

int tensor_fill(tensor* s, tensor* v, tensor* result) {

    if ((s  ==  NULL)  ||  (v  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(s);
    ASSERT_NUMERIC(v);

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

    int s_values = result->rows * result->columns;
    int v_len = v->rows * v->columns;

    if (s_values % v_len != 0)  {
        return ERR_SHAPE_MISMATCH; 
    }

    int create_numeric_tensor_result = create_numeric_tensor(result, NULL, new_rows, new_cols);
    if (create_numeric_tensor_result  !=  ERR_SUCCESS) {
        return create_numeric_tensor_result;
    }

    // OPTIMIZE
    for (int i = 0; i < s_values; i++) {
        int idx = i % v_len;
        result->values[i] = v->values[idx];
    }

    return ERR_SUCCESS;

}