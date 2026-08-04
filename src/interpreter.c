// Daniele Sterle SM3201594

#include "interpreter.h"

int interpret_file(const char* filename, tensor** stack, int* s_size, int* s_head) {
    FILE* file = fopen(filename, "r"); 
    if (file == NULL) {
        fprintf(stderr, "File: '%s'\n", filename); 
        return ERR_FILE_OPEN;
    }

    int curr;
    int last_char = ' ';
    int status = ERR_SUCCESS;

    while ((curr = getc(file)) != EOF) {
        // Skip whitespace
        if (isspace(curr)) {
            last_char = curr;
            continue;
        }

        // 1. Handle Tensor Parsing
        if (curr == '[') {
            status = parse_inline_tensor(file, stack, s_size, s_head, last_char);
            if (status != ERR_SUCCESS) break;
            last_char = ']';
            continue;
        }

        // 2. Handle String Parsing
        if (curr == '"') {
            status = parse_string_literal(file, stack, s_size, s_head);
            if (status != ERR_SUCCESS) break;
            last_char = '"';
            continue;
        }

        // 3. Handle Operations
        status = execute_operator(curr, stack, s_size, s_head);
        if (status != ERR_SUCCESS) break;
        
        last_char = curr;
    }

    fclose(file);
    return status;
}

int parse_inline_tensor(FILE* file, tensor** stack, int* s_size, int* s_head, int last_char) {
    if (last_char == ']' || last_char == '"') {
        return ERR_SYNTAX;
    }
    
    float* values = (float*) malloc(sizeof(float) * INITIAL_TENSOR_CAPACITY);
    if (values == NULL) {
        return ERR_OUT_OF_MEMORY;
    }

    int v_idx = 0;
    int v_size = INITIAL_TENSOR_CAPACITY;
    char buffer[MAX_FLOAT_STR_LEN];
    int b_idx = 0;
    int curr;
    
    last_char = '[';

    while ((curr = getc(file)) != ']') {
        
        if (curr == EOF) {
            free(values);
            return ERR_UNEXPECTED_EOF;
        }

        if (last_char == '[' && curr != ' ') {
            free(values);
            return ERR_SYNTAX;
        }

        if (curr == ' ') {
            if (b_idx > 0) {
                // If b_idx > 0, we just finished reading a float
                buffer[b_idx] = '\0';
                char* endptr;
                float curr_float = (float) strtod(buffer, &endptr);

                // Check for invalid number formats (e.g., multiple dots)
                if (*endptr != '\0') {
                    free(values);
                    return ERR_INVALID_NUMBER;
                } 

                // Resize array if necessary
                if (v_idx >= v_size) {
                    v_size *= 2;
                    float* temp = (float*) realloc(values, sizeof(float) * v_size);
                    if (temp == NULL) {
                        free(values);
                        return ERR_OUT_OF_MEMORY;
                    }
                    values = temp;
                }
                values[v_idx++] = curr_float;
                b_idx = 0; // Reset buffer
            }
            last_char = curr;
            continue;
        }

        if (curr == '\n' || curr == '\t' || curr == '\r') {
            free(values);
            return ERR_SYNTAX;
        }

        if (isdigit(curr) || curr == '.' || curr == '-') {
            if (curr == '-' && last_char != ' ') {
                free(values);
                return ERR_SYNTAX;
            }
            if (b_idx < MAX_FLOAT_STR_LEN - 1) {
                buffer[b_idx++] = curr;
            } else {
                free(values);
                return ERR_SYNTAX; // Overflowing float buffer
            }
            last_char = curr;
            continue;
        }
        
        free(values);
        return ERR_INVALID_CHAR;
    }

    if (last_char != ' ' && last_char != '[') {
        free(values);
        return ERR_SYNTAX;
    }

    if (v_idx == 0) {
        free(values);
        return ERR_EMPTY_TENSOR;
    }

    // Shrink-to-fit the final array
    v_size = v_idx;
    float* temp = (float*) realloc(values, sizeof(float) * v_size);
    if (temp == NULL) {
        free(values);
        return ERR_OUT_OF_MEMORY;
    }
    values = temp;

    // Initialize and push to stack
    tensor new_tensor;
    int tensor_init_numeric_result = tensor_init_numeric(&new_tensor, values, 1, v_size);
    if (tensor_init_numeric_result != ERR_SUCCESS) {
        free(values);
        return tensor_init_numeric_result;
    }

    int push_result = push(stack, new_tensor, s_size, *s_head);
    if (push_result < 0) {
        free(values);
        return push_result; // push_result contains the error code
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int parse_string_literal(FILE* file, tensor** stack, int* s_size, int* s_head) {
    
    char buffer_file[MAX_FILENAME_LEN];
    int b_idx = 0;
    int curr;
    
    
    while ((curr = getc(file)) != '"') {
        
        if (curr == EOF) {
            return ERR_UNEXPECTED_EOF;
        }

        if (b_idx >= MAX_FILENAME_LEN) { 
            return ERR_BUFFER_OVERFLOW;
        }

        buffer_file[b_idx] = curr;
        b_idx++;
    }
    
    buffer_file[b_idx] = '\0';
    
    // Initialize and push to stack
    tensor new_tensor;
    int tensor_init_string_result = tensor_init_string(&new_tensor, buffer_file);
    if (tensor_init_string_result != ERR_SUCCESS) {
        return tensor_init_string_result;
    }

    int push_result = push(stack, new_tensor, s_size, *s_head);
    if (push_result < 0) {
        return push_result; // push_result contains the error code
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int execute_operator(int op_char, tensor** stack, int* s_size, int* s_head) {
    switch (op_char) {
        
        /* Operazioni aritmetiche */
        case '+':
            return handle_add_op(stack, s_size, s_head);
            break;
        case '-':
            return handle_subtract_op(stack, s_size, s_head);
            break;
        case '*':
            return handle_multiply_op(stack, s_size, s_head);
            break;

        /* Operazioni di comparazione */
        case '<':
            return handle_lt_op(stack, s_size, s_head);
            break;
        case '>':
            return handle_gt_op(stack, s_size, s_head);
            break;
        case '=':
            return handle_eq_op(stack, s_size, s_head);
            break;

        /* Operazioni logiche */
        case '&':
            return handle_and_op(stack, s_size, s_head);
            break;
        case '|':
            return handle_or_op(stack, s_size, s_head);
            break;
        case '!':
            return handle_not_op(stack, s_size, s_head);
            break;

        /* Operazioni di selezione */
        case '$':
            return handle_select_op(stack, s_size, s_head);
            break;

        /* Operazioni specifiche per tensori */
        case '@':
            return handle_matmul_op(stack, s_size, s_head);
            break;
        case '.':
            return handle_dot_op(stack, s_size, s_head);
            break;
        case 'c':
            return handle_conv2d_op(stack, s_size, s_head);
            break;

        /* Operazioni sulla forma dei tensori */
        case 'r':
            return handle_reshape_op(stack, s_size, s_head);
            break;
        case '_':
            return handle_ravel_op(stack, s_size, s_head);
            break;
        case '#':
            return handle_get_shape_op(stack, s_size, s_head);
            break;

        /* Operazioni di generazione di numeri casuali */
        case '?':
            return handle_random_op(stack, s_size, s_head);
            break;

        /* Operazioni elemento per elemento */
        case 'R':
            return handle_relu_op(stack, s_size, s_head);
            break;
        case 'm':
            return handle_min_op(stack, s_size, s_head);
            break;
        case 'M':
            return handle_max_op(stack, s_size, s_head);
            break;

        /* Operazioni di riduzione */
        case 'S':
            return handle_sum_reduce_op(stack, s_size, s_head);
            break;

        /* Operazioni di filling di tensori */
        case 'f':
            return handle_fill_op(stack, s_size, s_head);
            break;

        /* Operazioni di utilità */
        case 'p':
            return handle_print_op(stack, s_head);
            break;

        /* Operazioni di manipolazione dello stack */
        case 'd':
            return stack_dup(stack, s_size, s_head);
            break;
        case 's':
            return stack_swap(stack, s_size, s_head);
            break;
        case 'o':
            return stack_over(stack, s_size, s_head);
            break;
        case 'D':
            return stack_drop(stack, s_head);
            break;

        /* I/O operations */
        case '(':
            return handle_read_pgm_op(stack, s_size, s_head);
            break;
        case ')':
            return handle_write_pgm_op(stack, s_head);
            break;
        case '{':
            return handle_read_mmap_op(stack, s_size, s_head);
            break;
        case '}':
            return handle_write_bin_op(stack, s_head);
            break;
        
        default:
            return ERR_SYNTAX;
            break;

        }
}