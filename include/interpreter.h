// Daniele Sterle SM3201594
#ifndef TENSORFORTH_INTERPRETER_H
#define TENSORFORTH_INTERPRETER_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "utils.h"
#include "ops_handler.h"
#include "ops_math.h"
#include "ops_tensor.h"
#include "stack.h"

#define INITIAL_TENSOR_CAPACITY 64
#define MAX_FLOAT_STR_LEN 64
#define MAX_FILENAME_LEN 256

int interpret_file(const char* filename, tensor** stack, int* s_size, int* s_head);

int parse_inline_tensor(FILE* file, tensor** stack, int* s_size, int* s_head, int last_char);
int parse_string_literal(FILE* file, tensor** stack, int* s_size, int* s_head);
int execute_operator(int op_char, tensor** stack, int* s_size, int* s_head);

#endif