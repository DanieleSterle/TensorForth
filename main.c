// Daniele Sterle SM3201594

// Application entry point and parsing.
// Reads the source file passed via CLI, tokenizes the input, and routes tokens.

#ifndef include
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include "stack.h"
#include "utils.h"
#include "ops_handler.h"
#include "ops_math.h"
#include "ops_tensor.h"
#define BUFFER_SIZE 64
#endif

int main(int argc, char *argv[]) {

    srand(time(NULL));

    if (argc < 2){
        print_error(ERR_MISSING_ARGUMENT);
        return ERR_MISSING_ARGUMENT;
    }

    tensor* stack = NULL;
    int s_size = create_stack(&stack);
    int s_head = 0;

    if (s_size  ==  -1) {
        print_error(ERR_OUT_OF_MEMORY);
        return ERR_OUT_OF_MEMORY;
    }

    FILE* file = fopen(argv[1], "r"); 
    int curr;
    if (file  ==  NULL) {
        fprintf(stderr, "File: '%s'\n", argv[1]); 
        print_error(ERR_FILE_OPEN);
        free_all(NULL, stack, s_head, NULL);
        return 1;
    }

    int last_char = ' ';

    while ((curr = getc(file))  !=  EOF) {
        
        // Gestione numero fuori posto
        if (isdigit(curr)) {
            print_error(ERR_SYNTAX);
            free_all(file, stack, s_head, NULL);
            return ERR_SYNTAX;
        }
        
        if (isspace(curr)) {
            last_char = curr; // Aggiorna lo stato prima di saltare
            continue;
        }

        if (curr == '['){

            if (last_char  ==  ']') {
                print_error(ERR_SYNTAX);
                free_all(file, stack, s_head, NULL);
                return ERR_SYNTAX;
            }

            if (last_char  ==  '"') {
                print_error(ERR_SYNTAX);
                free_all(file, stack, s_head, NULL);
                return ERR_SYNTAX;
            }
            
            float* values = (float*) malloc(sizeof(float) * DEF_VALUES_SIZE);
            int v_idx = 0;
            int v_size = DEF_VALUES_SIZE;

            char buffer[BUFFER_SIZE];
            int b_idx = 0;
            
            last_char = '[';

            while((curr = getc(file))  !=  ']'){
                
                if (curr == EOF) {
                    print_error(ERR_UNEXPECTED_EOF);
                    free_all(file, stack, s_head, values);
                    return ERR_UNEXPECTED_EOF;
                }

                if (last_char == '[' && curr != ' ') {
                    print_error(ERR_SYNTAX);
                    free_all(file, stack, s_head, values);
                    return ERR_SYNTAX;
                }

                if (curr  ==  ' '){
                    if (b_idx > 0){
                        // If b_idx > 0, it means we just finished reading a float
                        buffer[b_idx] = '\0';
                        //float curr_float = atof(buffer);

                        char* endptr;
                        float curr_float = (float)strtod(buffer, &endptr);

                        // Check if there are multiple . in a row (eg 10..5)
                        if (*endptr != '\0') {
                            print_error(ERR_INVALID_NUMBER);
                            free_all(file, stack, s_head, values);
                            return ERR_INVALID_NUMBER;
                        } 

                        // --- INSERIRE IN VALUES + RESIZE ---

                        if (v_idx >= v_size){
                            v_size *= 2;
                            float* temp = (float*) realloc(values, sizeof(float) * v_size);
    
                            // 3. Check if realloc failed
                            if (temp  ==  NULL) {
                                print_error(ERR_OUT_OF_MEMORY);
                                free_all(file, stack, s_head, values);
                                return ERR_OUT_OF_MEMORY;
                            }
                            
                            // 4. It succeeded! Update the main pointer
                            values = temp;
                        }
                        values[v_idx++] = curr_float;
                
                        b_idx = 0; // Reset buffer for the next float
                    }

                    // If b_idx is 0, it means this is a consecutive space.
                    // We just let the loop 'continue', completely ignoring it!
                    last_char = curr;
                    continue;
                }

                if ((curr  ==  '\n')  ||  (curr  ==  '\t')  ||  (curr  ==  '\r')) {
                    print_error(ERR_SYNTAX);
                    free_all(file, stack, s_head, values);
                    return ERR_SYNTAX;
                }

                if (isdigit(curr)  ||  curr  ==  '.'  ||  curr  ==  '-'){
                    
                    if (curr == '-' && last_char != ' ') {
                        print_error(ERR_SYNTAX);
                        free_all(file, stack, s_head, values);
                        return ERR_SYNTAX;
                    }

                    if (b_idx < BUFFER_SIZE - 1){
                        buffer[b_idx++] = curr;
                    }else{
                        print_error(ERR_SYNTAX);
                        free_all(file, stack, s_head, values);
                        return ERR_SYNTAX;
                    }
                    last_char = curr;
                    continue;
                }
                
                print_error(ERR_INVALID_CHAR);
                free_all(file, stack, s_head, values);
                return ERR_INVALID_CHAR;

            }

            if (last_char != ' ' && last_char != '[') {
                print_error(ERR_SYNTAX);
                free_all(file, stack, s_head, values);
                return ERR_SYNTAX;
            }

            if (v_idx == 0) {
                print_error(ERR_EMPTY_TENSOR);
                free_all(file, stack, s_head, values);
                return ERR_EMPTY_TENSOR;
            }

            // SHRINKING MALLOC LOGIC
            //int old_v_size = v_size;
            v_size = v_idx;
            float* temp = (float*) realloc(values, sizeof(float) * v_size);
    
            // 3. Check if realloc failed
            if (temp  ==  NULL) {
                print_error(ERR_OUT_OF_MEMORY);
                free_all(file, stack, s_head, values);
                return ERR_OUT_OF_MEMORY;
            }
            // 4. It succeeded! Update the main pointer
            values = temp;

            //printf("ho shrinkato correttamente da %d a %d", old_v_size, v_size);

            tensor new_tensor;
            int create_numeric_tensor_result = create_numeric_tensor(&new_tensor, values, 1, v_size);
            if (create_numeric_tensor_result  !=  0) {
                print_error(create_numeric_tensor_result);
                free_all(file, stack, s_head, values);
                return create_numeric_tensor_result;
            }

            s_head = push(&stack, new_tensor, &s_size, s_head);
            
            // MIGLIORARE
            if (s_head  ==  -1) {
                print_error(ERR_STACK_OVERFLOW);
                free_all(file, stack, s_head, values);
                return ERR_STACK_OVERFLOW;
            }

            //DEBUG
            /*printf("\nho fatto il push di:");
            printf("\n[");

            // Loop up to v_idx (the number of floats in this tensor), not s_head
            for (int i = 0; i < v_idx; i++){
                printf(" %.2f ", values[i]);
            }

            printf("]\n");
            */

            last_char = ']';
            continue;
        }

        if (curr == '"'){

            char buffer_file[512];
            int b_idx = 0;
            last_char = '(';
            
            while((curr = getc(file))  !=  '"'){
                
                if (curr  ==  EOF) {
                    print_error(ERR_UNEXPECTED_EOF);
                    free_all(file, stack, s_head, NULL);
                    return ERR_UNEXPECTED_EOF;
                }

                if (b_idx >= BUFFER_SIZE) { 
                    // Handle error: filename too long
                    print_error(ERR_BUFFER_OVERFLOW);
                    free_all(file, stack, s_head, NULL);
                    return ERR_BUFFER_OVERFLOW;
                }

                last_char = curr;
                buffer_file[b_idx] = curr;
                b_idx++;
            }
            
            last_char = curr;
            buffer_file[b_idx] = '\0';
            
            tensor new_tensor;
            int create_string_tensor_result = create_string_tensor(&new_tensor, buffer_file);
            if (create_string_tensor_result  !=  0) {
                print_error(create_string_tensor_result);
                free_all(file, stack, s_head, NULL);
                return create_string_tensor_result;
            }

            s_head = push(&stack, new_tensor, &s_size, s_head);
            
            // MIGLIORARE
            if (s_head  <  0) {
                print_error(ERR_STACK_OVERFLOW);
                free_all(file, stack, s_head, NULL);
                return ERR_STACK_OVERFLOW;
            }

            // printf("Ho fatto il push di Filename read: %s\n", buffer_file);
            continue;
        }

        // int status; - quando tutte operazioni implementate
        // DEBUG
        int status = ERR_SUCCESS;

        switch (curr) {
        
        /* Operazioni aritmetiche */
        case '+':
            status = handle_add_op(&stack, &s_size, &s_head);
            break;
        case '-':
            status = handle_subtract_op(&stack, &s_size, &s_head);
            break;
        case '*':
            status = handle_multiply_op(&stack, &s_size, &s_head);
            break;

        /* Operazioni di comparazione */
        case '<':
            status = handle_lt_op(&stack, &s_size, &s_head);
            break;
        case '>':
            status = handle_gt_op(&stack, &s_size, &s_head);
            break;
        case '=':
            status = handle_eq_op(&stack, &s_size, &s_head);
            break;

        /* Operazioni logiche */
        case '&':
            status = handle_and_op(&stack, &s_size, &s_head);
            break;
        case '|':
            status = handle_or_op(&stack, &s_size, &s_head);
            break;
        case '!':
            status = handle_not_op(&stack, &s_size, &s_head);
            break;

        /* Operazioni di selezione */
        case '$':
            status = handle_select_op(&stack, &s_size, &s_head);
            break;

        /* Operazioni specifiche per tensori */
        case '@':
            status = handle_matmul_op(&stack, &s_size, &s_head);
            break;
        case '.':
            status = handle_dot_op(&stack, &s_size, &s_head);
            break;
        case 'c':
            status = handle_conv2d_op(&stack, &s_size, &s_head);
            break;

        /* Operazioni sulla forma dei tensori */
        case 'r':
            status = handle_reshape_op(&stack, &s_size, &s_head);
            break;
        case '_': /* Note: Escaped backslash for C char literal */
            status = handle_ravel_op(&stack, &s_size, &s_head);
            break;
        case '#':
            status = handle_get_shape_op(&stack, &s_size, &s_head);
            break;

        /* Operazioni di generazione di numeri casuali */
        case '?':
            status = handle_random_op(&stack, &s_size, &s_head);
            break;

        /* Operazioni elemento per elemento */
        case 'R':
            status = handle_relu_op(&stack, &s_size, &s_head);
            break;
        case 'm':
            status = handle_min_op(&stack, &s_size, &s_head);
            break;
        case 'M':
            status = handle_max_op(&stack, &s_size, &s_head);
            break;

        /* Operazioni di riduzione */
        case 'S':
            status = handle_sum_reduce_op(&stack, &s_size, &s_head);
            break;

        /* Operazioni di filling di tensori */
        case 'f':
            status = handle_fill_op(&stack, &s_size, &s_head);
            break;

        /* Operazioni di utilità */
        case 'p':
            status = handle_print_op(&stack, &s_head);
            break;

        /* Operazioni di manipolazione dello stack */
        case 'd':
            status = stack_dup(&stack, &s_size, &s_head);
            break;
        case 's':
            status = stack_swap(&stack, &s_size, &s_head);
            break;
        case 'o':
            status = stack_over(&stack, &s_size, &s_head);
            break;
        case 'D':
            status = stack_drop(&stack, &s_head);
            break;

        /* I/O operations */
        case '(':
            status = handle_read_pgm_op(&stack, &s_size, &s_head);
            break;
        case ')':
            status = handle_write_pgm_op(&stack, &s_head);
            break;
        case '{':
            status = handle_read_mmap_op(&stack, &s_size, &s_head);
            break;
        case '}':
            status = handle_write_bin_op(&stack, &s_head);
            break;
        
        default:
            status = ERR_SYNTAX;
            break;

        }

        if (status != ERR_SUCCESS) {
            print_error(status);
            return status;
        }
    }

    free_all(file, stack, s_head, NULL);
    return 0;
}