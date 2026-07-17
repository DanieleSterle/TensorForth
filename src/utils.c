// Daniele Sterle SM3201594

// Helpers, printing, and randomness.
// Implements the print function p, random tensor generation ?, and error handling.

#include <stdlib.h>
#include "utils.h"

void free_all(FILE* file, tensor* stack, int s_stack, float* values) {

    if (file  !=  NULL) {
        close(file);
    }

    free_stack(stack, s_stack);    
    free(values);

}