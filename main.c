// Daniele Sterle SM3201594

// Application entry point and parsing.
// Reads the source file passed via CLI, tokenizes the input, and routes tokens.

#ifndef include
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "stack.h"
#include "utils.h"
#include "ops_handler.h"
#include "ops_math.h"
#define BUFFER_SIZE 64
#endif

// COME GESTIRE CODICI ERRORI INIZIALI?
int main(int argc, char *argv[]) {

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
        // Clean up the stack since it was allocated!
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

            last_char = ']';

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
            if (create_tensor(&new_tensor, values, 1, v_size)  ==  -1) {
                print_error(ERR_OUT_OF_MEMORY); // Or custom ERR_TENSOR_CREATION
                free_all(file, stack, s_head, values);
                return ERR_OUT_OF_MEMORY;
            }

            s_head = push(&stack, new_tensor, &s_size, s_head);
            
            if (s_head  ==  -1) {
                print_error(ERR_STACK_OVERFLOW);
                free_all(file, stack, s_head, values);
                return ERR_STACK_OVERFLOW;
            }

            //DEBUG
            printf("\nho fatto il push di:");
            printf("\n[");

            // Loop up to v_idx (the number of floats in this tensor), not s_head
            for (int i = 0; i < v_idx; i++){
                printf(" %.2f ", values[i]);
            }

            printf("]\n");

        }

        if (curr == '"'){
            //TODO: gestione file 
        }

        int status;

        // SOLO CHIAMARE FUNZIONI + print errori, NO BUSINESS LOGIC
        switch (curr) {
        
        /* Operazioni aritmetiche */
        case '+':

            status = handle_add_op(&stack, &s_size, &s_head);

            if (status != ERR_SUCCESS) {
                print_error(status); 
                return status;
            }else {


                // --- DEBUG ---
                printf("ho fatto la + e ho fatto push\n");
                printf("Stack head: %d\n", s_head);

                printf("\n[");

                for (int i = 0; i < stack[s_head - 1].columns * stack[s_head - 1].rows; i++){
                    printf(" %.2f ", stack[s_head - 1].values[i]);
                }

                printf("]\n");

            }
             
            break;
        case '-':
            
            status = handle_subtract_op(&stack, &s_size, &s_head);

            if (status != ERR_SUCCESS) {
                print_error(status); 
                return status;
            }else {


                // --- DEBUG ---
                printf("ho fatto la - e ho fatto push\n");
                printf("Stack head: %d\n", s_head);

                printf("\n[");

                for (int i = 0; i < stack[s_head - 1].columns * stack[s_head - 1].rows; i++){
                    printf(" %.2f ", stack[s_head - 1].values[i]);
                }

                printf("]\n");
            }

            break;
        case '*':
            
            status = handle_multiply_op(&stack, &s_size, &s_head);

            if (status != ERR_SUCCESS) {
                print_error(status); 
                return status;
            }else {


                // --- DEBUG ---
                printf("ho fatto la * e ho fatto push\n");
                printf("Stack head: %d\n", s_head);

                printf("\n[");

                for (int i = 0; i < stack[s_head - 1].columns * stack[s_head - 1].rows; i++){
                    printf(" %.2f ", stack[s_head - 1].values[i]);
                }

                printf("]\n");

            }

            break;

        /* Operazioni di comparazione */
        case '<':
            
            status = handle_lt_op(&stack, &s_size, &s_head);

            if (status != ERR_SUCCESS) {
                print_error(status); 
                return status;
            }else {


                // --- DEBUG ---
                printf("ho fatto la * e ho fatto push\n");
                printf("Stack head: %d\n", s_head);

                printf("\n[");

                for (int i = 0; i < stack[s_head - 1].columns * stack[s_head - 1].rows; i++){
                    printf(" %.2f ", stack[s_head - 1].values[i]);
                }

                printf("]\n");

            }

            break;
        case '>':
            

            status = handle_gt_op(&stack, &s_size, &s_head);

            if (status != ERR_SUCCESS) {
                print_error(status); 
                return status;
            }else {


                // --- DEBUG ---
                printf("ho fatto la * e ho fatto push\n");
                printf("Stack head: %d\n", s_head);

                printf("\n[");

                for (int i = 0; i < stack[s_head - 1].columns * stack[s_head - 1].rows; i++){
                    printf(" %.2f ", stack[s_head - 1].values[i]);
                }

                printf("]\n");

            }

            break;
        case '=':
            
            status = handle_eq_op(&stack, &s_size, &s_head);

            if (status != ERR_SUCCESS) {
                print_error(status); 
                return status;
            }else {


                // --- DEBUG ---
                printf("ho fatto la * e ho fatto push\n");
                printf("Stack head: %d\n", s_head);

                printf("\n[");

                for (int i = 0; i < stack[s_head - 1].columns * stack[s_head - 1].rows; i++){
                    printf(" %.2f ", stack[s_head - 1].values[i]);
                }

                printf("]\n");

            }


            break;

        /* Operazioni logiche */
        case '&':
            /* code */
            break;
        case '|':
            /* code */
            break;
        case '!':
            /* code */
            break;

        /* Operazioni di selezione */
        case '$':
            /* code */
            break;

        /* Operazioni specifiche per tensori */
        case '@':
            /* code */
            break;
        case '.':
            /* code */
            break;
        case 'c':
            /* code */
            break;

        /* Operazioni sulla forma dei tensori */
        case 'r':
            /* code */
            break;
        case '\\': /* Note: Escaped backslash for C char literal */
            /* code */
            break;
        case '#':
            /* code */
            break;

        /* Operazioni di generazione di numeri casuali */
        case '?':
            /* code */
            break;

        /* Operazioni elemento per elemento */
        case 'R':
            /* code */
            break;
        case 'm':
            /* code */
            break;
        case 'M':
            /* code */
            break;

        /* Operazioni di riduzione */
        case 'S':
            /* code */
            break;

        /* Operazioni di filling di tensori */
        case 'f':
            /* code */
            break;

        /* Operazioni di utilità */
        case 'p':
            /* code */
            break;

        /* Operazioni di manipolazione dello stack */
        case 'd':
            /* code */
            break;
        case 's':
            /* code */
            break;
        case 'o':
            /* code */
            break;
        case 'D':
            /* code */
            break;

        /* I/O operations */
        case '(':
            /* code */
            break;
        case ')':
            /* code */
            break;
        case '{':
            /* code */
            break;
        case '}':
            /* code */
            break;
        
        default:
            break;

        }
    }

    free_all(file, stack, s_head, NULL);
    return 0;
}