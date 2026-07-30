// Daniele Sterle SM3201594

#ifndef TENSORFORTH_STACK_H
#define TENSORFORTH_STACK_H

#define DEF_STACK_SIZE 4
#define DEF_VALUES_SIZE 64

typedef struct {
    // Row - Major
    float* values;
    int rows;
    int columns;
    int* ref_count;
} tensor;

int create_stack(tensor** stack);
int push(tensor** stack, tensor t, int* curr_stack_size, int idx_head);
int pop(tensor* stack, tensor* t, int idx_head);
int create_tensor(tensor* t, float* values, int rows, int columns);
void free_stack(tensor* stack, int idx_head);

int stack_dup(tensor** stack, int* s_size, int* s_head);
int stack_swap(tensor** stack, int* s_size, int* s_head);
int stack_over(tensor** stack, int* s_size, int* s_head);
int stack_drop(tensor** stack, int* s_head);

#endif