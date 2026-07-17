// Daniele Sterle SM3201594

// Helpers, printing, and randomness.
// Implements the print function p, random tensor generation ?, and error handling.

#include <stdlib.h>
#include "utils.h"

void free_all(FILE* file, tensor* stack, int s_stack, float* values) {

    if (file  !=  NULL) {
        fclose(file);
    }

    free_stack(stack, s_stack);    
    free(values);
}

int shape_cmp(tensor* t1, tensor* t2) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)) {
        return ERR_NULL_PTR; 
    }

    if ((t1->rows  ==  t2->rows)  &&  (t1->columns  ==  t2->columns)){
        return ERR_SUCCESS;
    }

    return ERR_SHAPE_MISMATCH;

}

void print_error(int error_code) {
    switch (error_code) {
        case ERR_SUCCESS:
            break;
        case ERR_NULL_PTR:
            fprintf(stderr, "ERRORE: Rilevato puntatore nullo.\n");
            break;
        case ERR_SHAPE_MISMATCH:
            fprintf(stderr, "ERRORE: Le dimensioni dei tensori non corrispondono per questa operazione.\n");
            break;
        case ERR_OUT_OF_MEMORY:
            fprintf(stderr, "ERRORE: Memoria esaurita (allocazione fallita).\n");
            break;
        case ERR_STACK_UNDERFLOW:
            fprintf(stderr, "ERRORE: Non ci sono abbastanza operandi nello stack.\n");
            break;
        case ERR_STACK_OVERFLOW:
            fprintf(stderr, "ERRORE: Capacità dello stack superata.\n");
            break;
        default:
            fprintf(stderr, "ERRORE: Si è verificato un errore sconosciuto (Codice: %d).\n", error_code);
            break;
    }
}
