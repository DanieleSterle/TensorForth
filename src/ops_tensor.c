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

    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, t1->shape[0], t2->shape[1]);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }
    
    result->ndim = 2;
    result->shape[0] = t1->shape[0];
    result->shape[1] = t2->shape[1];

    // OPTIMIZE WITH OPENMP & INDEXES
    // t2 should be in COLUMN MAJOR ?
    for (int i = 0; i < result->shape[0]; i++){
        for (int j = 0; j < result->shape[1]; j++){
            float sum = 0.0f;
            
            for (int k = 0; k < t1->shape[1]; k++) {
                // t1 index: row i, column k
                // t2 index: row k, column j
                sum += t1->values[i * t1->shape[1] + k] * t2->values[k * t2->shape[1] + j];
            }
            
            result->values[i * result->shape[1] + j] = sum;
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

    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, 1, 0);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }
    
    result->ndim = 1;
    result->shape[0] = 1;
    result->shape[1] = 0;

    // OPTIMIZE WITH OPENMP  
    float sum = 0.0f;
    int vector_length = t1->shape[0]; 
    
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

    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, t->shape[0], t->shape[1]);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    result->ndim = 2;
    result->shape[0] = t->shape[0];
    result->shape[1] = t->shape[1];

    int offset_row = k->shape[0] / 2;
    int offset_col = k->shape[1] / 2;

    // OPTIMIZE WITH OPENMP e memoria
    for (int i = 0; i < result->shape[0]; i++) {
        for (int j = 0; j < result->shape[1]; j++) {
            
            float sum = 0.0f;

            for (int x = 0; x < k->shape[0]; x++) {
                for (int y = 0; y < k->shape[1]; y++) {
                    
                    int curr_idx_row = i + x - offset_row;
                    int curr_idx_column = j + y - offset_col;

                    if ((curr_idx_row >= 0)  &&  (curr_idx_column >= 0)  &&
                        (curr_idx_row < t->shape[0])  &&  (curr_idx_column < t->shape[1])){
                        
                        int t_index = curr_idx_row * t->shape[1] + curr_idx_column;
                        int k_index = x * k->shape[1] + y;
                        
                        sum += t->values[t_index] * k->values[k_index];
                    }
                }
            }
            
            result->values[i * result->shape[1] + j] = sum;
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

    int s_len = s->shape[0];

    if ((s_len < 1)  ||  (s_len > 2)) {
        return ERR_SHAPE_MISMATCH; 
    }

    int new_shape0;
    int new_shape1;
    int new_ndim;

    if (s_len == 2) {
        // shape is 2D Matrix
        new_shape0 = (int)s->values[0];
        new_shape1 = (int)s->values[1];
        new_ndim = 2;
    } else {
        // shape is 1D Vector 
        new_shape0 = (int)s->values[0];
        new_shape1 = 0;
        new_ndim = 1;
    }

    if (new_shape0 <= 0) {
        return ERR_SHAPE_MISMATCH;
    }
    
    if (new_ndim == 2) {
        if (new_shape1 <= 0) {
            return ERR_SHAPE_MISMATCH;
        }
    }

    int t_shape;
    if (t->ndim == 1) {
        t_shape = t->shape[0];
    } else {
        t_shape = t->shape[0] * t->shape[1];
    }
    
    int s_shape;
    if (new_ndim == 1) {
        s_shape = new_shape0;
    } else {
        s_shape = new_shape0 * new_shape1;
    }
    
    if (t_shape  !=  s_shape) {
        return ERR_SHAPE_MISMATCH;
    }

    t->ndim = new_ndim;
    t->shape[0] = new_shape0;
    t->shape[1] = new_shape1;

    return ERR_SUCCESS;
}

int tensor_ravel(tensor* t) {

    if (t  ==  NULL) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t);

    int t_shape;
    if (t->ndim == 1) {
        t_shape = t->shape[0];
    } else {
        t_shape = t->shape[0] * t->shape[1];
    }

    t->ndim = 1;
    t->shape[0] = t_shape;
    t->shape[1] = 0;

    return ERR_SUCCESS;
}

int tensor_get_shape(tensor* t, tensor* result) {

    if ((t  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t);

    int tensor_init_numeric_result;
    
    if (t->ndim == 1) {
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, 1, 0);
    } else {
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, 2, 0);
    }
    
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    result->ndim = 1;

    if (t->ndim == 1) {
        result->shape[0] = 1;
        result->shape[1] = 0;
        result->values[0] = (float)t->shape[0];
    } else {
        result->shape[0] = 2;
        result->shape[1] = 0;
        result->values[0] = (float)t->shape[0];
        result->values[1] = (float)t->shape[1];
    }

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

    int s_len = s->shape[0];

    if ((s_len < 1)  ||  (s_len > 2)) {
        return ERR_SHAPE_MISMATCH; 
    }

    int new_shape0;
    int new_shape1;
    int new_ndim;

    if (s_len == 2) {
        // shape is 2D Matrix
        new_shape0 = (int)s->values[0];
        new_shape1 = (int)s->values[1];
        new_ndim = 2;
    } else {
        // shape is 1D Vector 
        new_shape0 = (int)s->values[0];
        new_shape1 = 0;
        new_ndim = 1;
    }

    if (new_shape0 <= 0) {
        return ERR_SHAPE_MISMATCH;
    }
    
    if (new_ndim == 2) {
        if (new_shape1 <= 0) {
            return ERR_SHAPE_MISMATCH;
        }
    }

    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, new_shape0, new_shape1);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }
    
    result->ndim = new_ndim;
    result->shape[0] = new_shape0;
    result->shape[1] = new_shape1;

    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }
    
    int v_len;
    if (v->ndim == 1) {
        v_len = v->shape[0];
    } else {
        v_len = v->shape[0] * v->shape[1];
    }

    if (s_values % v_len != 0)  {
        return ERR_SHAPE_MISMATCH; 
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

    int s_len = s->shape[0];

    if ((s_len < 1)  ||  (s_len > 2)) {
        return ERR_SHAPE_MISMATCH; 
    }

    int new_shape0;
    int new_shape1;
    int new_ndim;

    if (s_len == 2) {
        // shape is 2D Matrix
        new_shape0 = (int)s->values[0];
        new_shape1 = (int)s->values[1];
        new_ndim = 2;
    } else {
        // shape is 1D Vector 
        new_shape0 = (int)s->values[0];
        new_shape1 = 0;
        new_ndim = 1;
    }

    if (new_shape0 <= 0) {
        return ERR_SHAPE_MISMATCH;
    }
    
    if (new_ndim == 2) {
        if (new_shape1 <= 0) {
            return ERR_SHAPE_MISMATCH;
        }
    }

    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, new_shape0, new_shape1);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }
    
    result->ndim = new_ndim;
    result->shape[0] = new_shape0;
    result->shape[1] = new_shape1;

    int size;
    if (new_ndim == 1) {
        size = new_shape0;
    } else {
        size = new_shape0 * new_shape1;
    }

    // NON OTTIMIZZARE
    for (int i = 0; i < size; i++){
        result->values[i] = (float)rand() / (float)RAND_MAX;
    }

    return ERR_SUCCESS;

}