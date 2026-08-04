// Daniele Sterle SM3201594

#ifndef TENSORFORTH_TENSOR_H
#define TENSORFORTH_TENSOR_H

#include <stdint.h>
#include <sys/types.h>
#include <string.h>
#include <sys/mman.h>
#include "utils.h"

#define MAX_DIM 2

typedef enum {
    TYPE_NUMERIC,
    TYPE_NUMERIC_MMAP,
    TYPE_STRING
} tensor_type;

typedef struct {
    // The tag to check what data is active
    tensor_type type;
    
    union {                 
        
        struct {            
            float* values;
            // e.g., shape[0] for 1D; shape[0] (rows), shape[1] (cols) for 2D
            int32_t shape[MAX_DIM];
            // 1 for vector, 2 for matrix
            int32_t ndim;
        };                  
        
        char* filename;
    };
    
    int* ref_count;
} tensor;

typedef struct {
    int32_t shape[MAX_DIM];
    int32_t ndim;
    off_t data_offset;
} on_disk_tensor;

int shape_cmp(tensor* t1, tensor* t2);
int shape_cmp_matmul(tensor* t1, tensor* t2);
int shape_cmp_dot(tensor* t1, tensor* t2);
int is_boolean(tensor* t);
int is_matrix(tensor* t);
int is_vector(tensor* t);

int tensor_init_numeric(tensor* t, float* values, int rows, int columns);
int tensor_init_string(tensor* t, char* string);
void free_tensor(tensor* t);

void tensor_print(tensor* t);

#endif