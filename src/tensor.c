// Daniele Sterle SM3201594

// Tensor data structures and lifecycle.
// Handles memory allocation, reference counting, and shape operations like r, \, and #.

#include "tensor.h"

int shape_cmp(tensor* t1, tensor* t2) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)) {
        return ERR_NULL_PTR; 
    }

    if (t1->ndim == t2->ndim) {
        if (t1->ndim == 1) {
            if (t1->shape[0] == t2->shape[0]) {
                return ERR_SUCCESS;
            }
        } else {
            if ((t1->shape[0] == t2->shape[0]) && (t1->shape[1] == t2->shape[1])) {
                return ERR_SUCCESS;
            }
        }
    }

    return ERR_SHAPE_MISMATCH;

}

int shape_cmp_matmul(tensor* t1, tensor* t2) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)) {
        return ERR_NULL_PTR; 
    }

    if (t1->shape[1]  ==  t2->shape[0]){
        return ERR_SUCCESS;
    }

    return ERR_MATMUL_DIM_MISMATCH;

}

int shape_cmp_dot(tensor* t1, tensor* t2) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)) {
        return ERR_NULL_PTR; 
    }

    int len_t1;
    if (t1->ndim == 1) {
        len_t1 = t1->shape[0];
    } else {
        len_t1 = t1->shape[0] * t1->shape[1];
    }
    
    int len_t2;
    if (t2->ndim == 1) {
        len_t2 = t2->shape[0];
    } else {
        len_t2 = t2->shape[0] * t2->shape[1];
    }

    if (len_t1  ==  len_t2) {
        return ERR_SUCCESS;
    }

    return ERR_DOT_DIM_MISMATCH; 
}

int is_boolean(tensor* t) {

    if (t  ==  NULL) return ERR_NULL_PTR; 

    int s_values;
    if (t->ndim == 1) {
        s_values = t->shape[0];
    } else {
        s_values = t->shape[0] * t->shape[1];
    }
    
    // OPTIMIZE 
    for (int i = 0; i < s_values; i++) {
       if ((t->values[i]  !=  0.0)  &&  (t->values[i]  !=  1.0)) return ERR_NOT_BOOLEAN;
    }

    return ERR_SUCCESS;

}

int is_matrix(tensor* t) {
    
    if (t  ==  NULL) return ERR_NULL_PTR;

    if (t->ndim == 2) {
        return ERR_SUCCESS;
    }

    return ERR_NOT_MATRIX;

}

int is_vector(tensor* t) {

    if (t  ==  NULL) return ERR_NULL_PTR;

    if (t->ndim == 1) {
        return ERR_SUCCESS;
    }

    return ERR_NOT_VECTOR;

}

int tensor_init_numeric(tensor* t, float* values, int rows, int columns) {

    if (t  ==  NULL) {
        return ERR_NULL_PTR; 
    }
    
    if (rows <= 0) {
        return ERR_SHAPE_MISMATCH; 
    }
    
    if (columns < 0) {
        return ERR_SHAPE_MISMATCH; 
    }

    t->type = TYPE_NUMERIC;
    
    int s_values;
    if (columns == 0) {
        t->ndim = 1;
        t->shape[0] = rows;
        t->shape[1] = 0;
        s_values = rows;
    } else {
        t->ndim = 2;
        t->shape[0] = rows;
        t->shape[1] = columns;
        s_values = rows * columns;
    }

    int newly_allocated = 0;

    if (values  ==  NULL) {
        float* temp = (float*) malloc(sizeof(float) * s_values);

        if (temp  ==  NULL) return ERR_OUT_OF_MEMORY;

        t->values = temp;
        newly_allocated = 1;

    } else {
        t->values = values;
    }

    t->ref_count = (int*) malloc(sizeof(int));
    
    if (t->ref_count  ==  NULL) {
        if (newly_allocated  ==  1) {
            free(t->values); 
        }
        return ERR_OUT_OF_MEMORY;
    }

    (*t->ref_count) = 1;

    return ERR_SUCCESS;
}

int tensor_init_string(tensor* t, char* string) {
    if ((t  ==  NULL)  ||  (string  ==  NULL)) {
        return ERR_NULL_PTR; 
    }
    
    t->type = TYPE_STRING;

    int str_len = strlen(string);
    t->filename = (char*) malloc(str_len + 1);
    
    if (t->filename == NULL) {
        return ERR_OUT_OF_MEMORY;
    }
    
    strcpy(t->filename, string);

    t->ref_count = (int*) malloc(sizeof(int));
    if (t->ref_count  ==  NULL) {
        return ERR_OUT_OF_MEMORY;
    }

    (*t->ref_count) = 1;

    return ERR_SUCCESS;
}

void free_tensor(tensor* t) {
    if ((t == NULL) || (t->ref_count == NULL)) {
        return;
    }

    (*t->ref_count)--;

    if ((*t->ref_count) == 0) {
        
        if (t->type == TYPE_NUMERIC) {
            if (t->values != NULL) {
                free(t->values);
            }
        } 
        
        if (t->type == TYPE_NUMERIC_MMAP) {
            if (t->values != NULL) {
                int total_elements;
                if (t->ndim == 1) {
                    total_elements = t->shape[0];
                } else {
                    total_elements = t->shape[0] * t->shape[1];
                }

                // 1. Calculate the exact total size of the file in memory
                size_t total_size = sizeof(on_disk_tensor) + 
                                   (total_elements * sizeof(float));

                // 2. Step backwards from t->values to find the original mmap pointer
                void* original_mmap_ptr = (void*)((char*)t->values - sizeof(on_disk_tensor));

                munmap(original_mmap_ptr, total_size);
            }
        } 
        
        if (t->type == TYPE_STRING) {
            if (t->filename != NULL) {
                free(t->filename);
            }
        }

        free(t->ref_count);
    }
}

void tensor_print(tensor* t) {
    
    if (t == NULL) {
        return;
    }

    if ((t->type  !=  TYPE_NUMERIC)  &&  (t->type  !=  TYPE_NUMERIC_MMAP)) return;

    // shape
    printf("Tensor(shape=[");
    if (t->ndim == 1) {
        printf("%d", t->shape[0]);
    } else {
        printf("%d %d", t->shape[0], t->shape[1]);
    }
    
    printf("], data=[");

    int total_elements;
    if (t->ndim == 1) {
        total_elements = t->shape[0];
    } else {
        total_elements = t->shape[0] * t->shape[1];
    }
    
    for (int i = 0; i < total_elements; i++) {
        printf("%.2f", t->values[i]);
        if (i < total_elements - 1) {
            printf(" ");
        }
    }
    
    printf("])\n");
}