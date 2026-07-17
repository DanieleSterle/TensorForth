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
} tensor;

int create_stack(tensor** stack);
int push(tensor** stack, tensor t, int* curr_stack_size, int idx_head);
int pop(tensor* stack, tensor* t, int idx_head);
int create_tensor(tensor* t, float* values, int rows, int columns);
void free_stack(tensor* stack, int idx_head);

// duplicate
// swap
// over
// drop

#endif