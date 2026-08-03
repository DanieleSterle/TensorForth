// Daniele Sterle SM3201594

// Application entry point and parsing.
// Reads the source file passed via CLI, tokenizes the input, and routes tokens.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "stack.h"
#include "utils.h"
#include "interpreter.h"

int main(int argc, char *argv[]) {
    srand(time(NULL));

    if (argc < 2){
        print_error(ERR_MISSING_ARGUMENT);
        return ERR_MISSING_ARGUMENT;
    }

    tensor* stack = NULL;
    int s_size = create_stack(&stack);
    int s_head = 0;

    if (s_size == ERR_OUT_OF_MEMORY) {
        print_error(ERR_OUT_OF_MEMORY);
        return ERR_OUT_OF_MEMORY;
    }

    // Hand off execution to the interpreter module
    int status = interpret_file(argv[1], &stack, &s_size, &s_head);

    if (status != ERR_SUCCESS) {
        print_error(status);
        free_stack(stack, s_head);
        return status;
    }

    free_stack(stack, s_head);
    return status;
}