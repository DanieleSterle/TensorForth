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

// Capacità iniziale predefinita per l'allocazione dei tensori
#define INITIAL_TENSOR_CAPACITY 64

// Lunghezza massima consentita per le stringhe che rappresentano numeri float
#define MAX_FLOAT_STR_LEN 64

// Lunghezza massima consentita per i nomi dei file
#define MAX_FILENAME_LEN 256

int interpret_file(const char* filename, tensor** stack, int* s_size, int* s_head);

int parse_inline_tensor(FILE* file, tensor** stack, int* s_size, int* s_head, int last_char);
int parse_string_literal(FILE* file, tensor** stack, int* s_size, int* s_head);
int execute_operator(int op_char, tensor** stack, int* s_size, int* s_head);

#endif