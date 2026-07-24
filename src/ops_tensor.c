// Daniele Sterle SM3201594

// Advanced tensor computations.
// Implements matrix multiplication @, dot product ., and 2D convolution c.

#include "ops_tensor.h"
#include "utils.h"

int tensor_matmul(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int matrix_result;
    matrix_result = is_matrix(t1);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    matrix_result = is_matrix(t2);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    int matmul_shape_result = shape_cmp_matmul(t1, t2);
    if (matmul_shape_result != ERR_SUCCESS) return matmul_shape_result;

    result->columns = t2->columns;
    result->rows = t1->rows;

    int s_values = result->rows * result->columns;
    result->values = malloc(sizeof(float) * s_values);

    if (result->values  ==  NULL) return ERR_OUT_OF_MEMORY; 

    // OPTIMIZE WITH OPENMP & INDEXES
    // t2 should be in COLUMN MAJOR


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

    int shape_result = shape_cmp_dot(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    result->columns = 1;
    result->rows = 1;
    result->values = malloc(sizeof(float));

    if (result->values  ==  NULL) return ERR_OUT_OF_MEMORY; 

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

    int matrix_result;
    matrix_result = is_matrix(t);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    matrix_result = is_matrix(k);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    result->columns = t->columns;
    result->rows = t->rows;

    int s_values = result->rows * result->columns;
    result->values = malloc(sizeof(float) * s_values);

    if (result->values  ==  NULL) return ERR_OUT_OF_MEMORY; 

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