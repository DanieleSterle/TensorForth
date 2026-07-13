// Daniele Sterle SM3201594

#ifndef var
#define DEF_STACK_SIZE 8
#define DEF_VALUES_SIZE 64
#include <stdio.h>
#endif


typedef struct {
    // Row - Major
    float* values;
    int rows;
    int columns;
} tensor;


tensor* create_stack();


int push(tensor* stack, tensor element, int curr_stack_size, int idx_head);
// Controlla se struct/stringa:
// - struct: push normale
// - stringa (operazione): pop operandi + push output operazione 
// Argomenti: stack, struct
// Return codice errore?


int pop(tensor* stack, int idx_head, tensor* out_element);
void free_stack(tensor* stack, int idx_head);

tensor create_tensor(float* values, int rows, int columns);

// duplicate
// swap 
// over
// drop