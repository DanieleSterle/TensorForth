// Daniele Sterle SM3201594

// Advanced tensor computations.
// Implements matrix multiplication @, dot product ., and 2D convolution c.

#include "ops_tensor.h"

int tensor_matmul(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int matrix_result;
    matrix_result = is_matrix(t1);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    matrix_result = is_matrix(t2);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    int matmul_shape_result = shape_cmp_matmul(t1, t2);
    if (matmul_shape_result != ERR_SUCCESS) return matmul_shape_result;

    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, t1->rows, t2->columns);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    // OPTIMIZE WITH OPENMP & INDEXES
    // t2 should be in COLUMN MAJOR ?
    for (int i = 0; i < result->rows; i++){
        for (int j = 0; j < result->columns; j++){
            float sum = 0.0f;
            
            for (int k = 0; k < t1->columns; k++) {
                // t1 index: row i, column k
                // t2 index: row k, column j
                sum += t1->values[i * t1->columns + k] * t2->values[k * t2->columns + j];
            }
            
            result->values[i * result->columns + j] = sum;
        }
    }

    return ERR_SUCCESS;
}

int tensor_dot(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int shape_result = shape_cmp_dot(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, 1, 1);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    // OPTIMIZE WITH OPENMP  
    float sum = 0.0f;
    int vector_length = t1->columns; 
    
    for (int k = 0; k < vector_length; k++) {
        sum += t1->values[k] * t2->values[k];
    }
    
    result->values[0] = sum;

    return ERR_SUCCESS;
}

int tensor_conv2d(tensor* t, tensor* k, tensor* result) {

    if ((t  ==  NULL)  ||  (k  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t);
    ASSERT_NUMERIC(k);

    int matrix_result;
    matrix_result = is_matrix(t);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    matrix_result = is_matrix(k);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, t->rows, t->columns);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int offset_row = k->rows / 2;
    int offset_col = k->columns / 2;

    // OPTIMIZE WITH OPENMP e memoria
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->columns; j++) {
            
            float sum = 0.0f;

            for (int x = 0; x < k->rows; x++) {
                for (int y = 0; y < k->columns; y++) {
                    
                    int curr_idx_row = i + x - offset_row;
                    int curr_idx_column = j + y - offset_col;

                    if ((curr_idx_row >= 0)  &&  (curr_idx_column >= 0)  &&
                        (curr_idx_row < t->rows)  &&  (curr_idx_column < t->columns)){
                        
                        int t_index = curr_idx_row * t->columns + curr_idx_column;
                        int k_index = x * k->columns + y;
                        
                        sum += t->values[t_index] * k->values[k_index];
                    }
                }
            }
            
            result->values[i * result->columns + j] = sum;
        }   
    }

    return ERR_SUCCESS;
}

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

    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, 1, 2);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
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

    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, new_rows, new_cols);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    // OPTIMIZE
    for (int i = 0; i < s_values; i++) {
        int idx = i % v_len;
        result->values[i] = v->values[idx];
    }

    return ERR_SUCCESS;

}

int tensor_generate_random(tensor* s, tensor* result) {
    if ((s  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int vector_result = is_vector(s);
    if (vector_result  !=  ERR_SUCCESS) return vector_result;

    int s_len = s->rows * s->columns;

    if ((s_len < 1)  ||  (s_len > 2)) {
        return ERR_SHAPE_MISMATCH; 
    }

    int rows;
    int cols;

    if (s_len == 2) {
        // shape is 2D Matrix
        rows = (int)s->values[0];
        cols = (int)s->values[1];
    } else {
        // shape is 1D Vector 
        rows = 1;
        cols = (int)s->values[0];
    }

    if ((rows <= 0)  ||  (cols <= 0)) {
        return ERR_SHAPE_MISMATCH;
    }

    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, rows, cols);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int size = cols * rows;

    // NON OTTIMIZZARE
    for (int i = 0; i < size; i++){
        result->values[i] = (float)rand() / (float)RAND_MAX;
    }

    return ERR_SUCCESS;

}