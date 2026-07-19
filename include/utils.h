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
    // Math Operations
    ERR_SHAPE_MISMATCH = -5,
    // I/O and Files
    ERR_MISSING_ARGUMENT = -6,
    ERR_FILE_OPEN = -7,
    ERR_UNEXPECTED_EOF = -8,
    // Syntax and Parsing
    ERR_SYNTAX = -9,
    ERR_INVALID_CHAR = -10,
    ERR_BUFFER_OVERFLOW = -11,
    ERR_EMPTY_TENSOR = -12,
    ERR_INVALID_NUMBER = -13
} error_code;

void free_all(FILE* fd, tensor* stack, int s_stack, float* values);
int shape_cmp(tensor* t1, tensor* t2);
void print_error(int error_code);

#endif