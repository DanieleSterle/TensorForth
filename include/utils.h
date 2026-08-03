// Daniele Sterle SM3201594

#ifndef TENSORFORTH_UTILS_H
#define TENSORFORTH_UTILS_H

#include <stdio.h>
#include <stdlib.h>

#define ASSERT_NUMERIC(t) \
    if ((t)->type != TYPE_NUMERIC) { \
        return ERR_EXPECTED_NUMERIC; \
    }

#define ASSERT_STRING(t) \
    if ((t)->type != TYPE_STRING) { \
        return ERR_EXPECTED_STRING; \
    }

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
    ERR_NOT_VECTOR = -10,
    ERR_EXPECTED_NUMERIC = -11,
    ERR_EXPECTED_STRING = -12,

    // I/O and Files
    ERR_MISSING_ARGUMENT = -13,
    ERR_FILE_OPEN = -14,
    ERR_UNEXPECTED_EOF = -15,
    ERR_FILE_READ = -16,      
    ERR_FILE_WRITE = -17,     
    ERR_INVALID_PGM = -18,    
    ERR_FILE_STAT = -19,      
    ERR_MMAP_FAILED = -20,    
    ERR_IO = -21,             

    // Syntax and Parsing
    ERR_SYNTAX = -22,
    ERR_INVALID_CHAR = -23,
    ERR_BUFFER_OVERFLOW = -24,
    ERR_EMPTY_TENSOR = -25,
    ERR_INVALID_NUMBER = -26
} error_code;

void print_error(int error_code);

#endif