// Daniele Sterle SM3201594

#ifndef TENSORFORTH_UTILS_H
#define TENSORFORTH_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include "stack.h"

typedef enum {
    ERR_SUCCESS = 0,
    // Memory & Stack
    ERR_NULL_PTR = -1,
    ERR_OUT_OF_MEMORY = -2,
    ERR_STACK_UNDERFLOW = -3,
    ERR_STACK_OVERFLOW = -4,
    // Math & Logic Operations
    ERR_SHAPE_MISMATCH = -5,
    ERR_NOT_BOOLEAN = -6,
    ERR_NOT_MATRIX = -7,
    ERR_DOT_DIM_MISMATCH = -8,
    ERR_MATMUL_DIM_MISMATCH = -9,
    // I/O and Files
    ERR_MISSING_ARGUMENT = -10,
    ERR_FILE_OPEN = -11,
    ERR_UNEXPECTED_EOF = -12,
    // Syntax and Parsing
    ERR_SYNTAX = -13,
    ERR_INVALID_CHAR = -14,
    ERR_BUFFER_OVERFLOW = -15,
    ERR_EMPTY_TENSOR = -16,
    ERR_INVALID_NUMBER = -17
} error_code;

void free_all(FILE* fd, tensor* stack, int s_stack, float* values);
int shape_cmp(tensor* t1, tensor* t2);
int shape_cmp_matmul(tensor* t1, tensor* t2);
int shape_cmp_dot(tensor* t1, tensor* t2);
int is_boolean(tensor* t);
int is_matrix(tensor* t);
//int is_vetcor(tensor* t);
void print_error(int error_code);

#endif