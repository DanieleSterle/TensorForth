// Daniele Sterle SM3201594

#ifndef TENSORFORTH_STACK_H
#define TENSORFORTH_STACK_H

#include <string.h>

#define DEF_STACK_SIZE 4
#define DEF_VALUES_SIZE 64

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
            int rows;
            int columns;
        };                  
        
        char* filename;
    };
    
    int* ref_count;
} tensor;

int create_stack(tensor** stack);
int push(tensor** stack, tensor t, int* curr_stack_size, int idx_head);
int pop(tensor* stack, tensor* t, int idx_head);
int create_numeric_tensor(tensor* t, float* values, int rows, int columns);
int create_string_tensor(tensor* t, char* string);
void free_stack(tensor* stack, int idx_head);

int stack_dup(tensor** stack, int* s_size, int* s_head);
int stack_swap(tensor** stack, int* s_size, int* s_head);
int stack_over(tensor** stack, int* s_size, int* s_head);
int stack_drop(tensor** stack, int* s_head);

#endif