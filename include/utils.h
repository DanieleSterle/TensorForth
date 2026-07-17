// Daniele Sterle SM3201594

#ifndef TENSORFORTH_UTILS_H
#define TENSORFORTH_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include "stack.h"

typedef enum {
    ERR_SUCCESS = 0,
    ERR_NULL_PTR = -1,
    ERR_SHAPE_MISMATCH = -2,
    ERR_OUT_OF_MEMORY = -3,
    ERR_STACK_UNDERFLOW = -4,
    ERR_STACK_OVERFLOW = -5
} error_code;

void free_all(FILE* fd, tensor* stack, int s_stack, float* values);
int shape_cmp(tensor* t1, tensor* t2);
void print_error(int error_code);

#endif