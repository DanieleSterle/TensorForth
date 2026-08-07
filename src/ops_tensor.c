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

    int shape[] = {t1->shape[0], t2->shape[1]};

    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    // 1. Initialize result to 0 (Required for i-k-j order)
    int res_elements = result->shape[0] * result->shape[1];
    #pragma omp parallel for
    for (int i = 0; i < res_elements; i++) {
        result->values[i] = 0.0f;
    }

    // 2. Compute using i-k-j order
    // Note: We cannot use collapse(2) here because i and j are no longer adjacent.
    #pragma omp parallel for
    for (int i = 0; i < result->shape[0]; i++){
        for (int k = 0; k < t1->shape[1]; k++) {
            
            // This value is constant for the entire j loop
            float t1_val = t1->values[i * t1->shape[1] + k];
            
            for (int j = 0; j < result->shape[1]; j++){
                // Innermost loop uses + j, which means perfect sequential memory access!
                result->values[i * result->shape[1] + j] += t1_val * t2->values[k * t2->shape[1] + j];
            }
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

    int vector_result;
    vector_result = is_vector(t1);
    if (vector_result != ERR_SUCCESS) return vector_result;

    vector_result = is_vector(t2);
    if (vector_result != ERR_SUCCESS) return vector_result;

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int shape[] = {1};
    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    float sum = 0.0f;
    int vector_length = t1->shape[0]; 
    
    #pragma omp parallel for reduction(+:sum)
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

    int shape[] = {t->shape[0], t->shape[1]};
    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int offset_row = k->shape[0] / 2;
    int offset_col = k->shape[1] / 2;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < result->shape[0]; i++) {
        for (int j = 0; j < result->shape[1]; j++) {
            
            float sum = 0.0f;

            // 1. Compute valid kernel bounds for the current pixel (i, j).
            // This mathematically crops the kernel at the image edges, completely 
            // eliminating the need for expensive "if" statements inside the inner loops.
            int start_x = (offset_row > i) ? (offset_row - i) : 0;
            int max_x   = t->shape[0] - i + offset_row;
            int end_x   = (k->shape[0] < max_x) ? k->shape[0] : max_x;

            int start_y = (offset_col > j) ? (offset_col - j) : 0;
            int max_y   = t->shape[1] - j + offset_col;
            int end_y   = (k->shape[1] < max_y) ? k->shape[1] : max_y;

            for (int x = start_x; x < end_x; x++) {
                
                int curr_idx_row = i + x - offset_row;
                
                // 2. Hoist invariant index math out of the innermost loop.
                // Pre-calculating the 1D base array offsets here prevents the CPU 
                // from doing redundant multiplication on every single 'y' iteration.
                int t_base_idx = curr_idx_row * t->shape[1] + j - offset_col;
                int k_base_idx = x * k->shape[1];

                // 3. Branchless multiply-accumulate over the valid row segment.
                // Because there are no 'if' checks, the compiler can safely translate 
                // this pure math into high-speed SIMD vector instructions (AVX).
                for (int y = start_y; y < end_y; y++) {
                    sum += t->values[t_base_idx + y] * k->values[k_base_idx + y];
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
        int shape[] = {1};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {2};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
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

    int tensor_shape;
    if (new_ndim == 1) {
        tensor_shape = TENSOR_SHAPE_VECTOR;
    } else {
        tensor_shape = TENSOR_SHAPE_MATRIX;
    }

    int shape[] = {new_shape0, new_shape1};
    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, tensor_shape);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

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

    #pragma omp parallel for
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

    int tensor_shape;
    if (new_ndim == 1) {
        tensor_shape = TENSOR_SHAPE_VECTOR;
    } else {
        tensor_shape = TENSOR_SHAPE_MATRIX;
    }

    int shape[] = {new_shape0, new_shape1};
    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, tensor_shape);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    // 1. Pre-calculate the division
    float scale = 1.0f / (float)RAND_MAX;
    
    // Grab a seed from the global generator before going parallel
    unsigned int base_seed = rand(); 

    #pragma omp parallel 
    {
        // Add the thread ID so each thread gets a unique starting point
        unsigned int seed = base_seed + omp_get_thread_num(); 

        #pragma omp for
        for (int i = 0; i < s_values; i++){
            result->values[i] = (float)rand_r(&seed) * scale;
        }
    }

    return ERR_SUCCESS;
}