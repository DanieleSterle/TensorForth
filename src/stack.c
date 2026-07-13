// Daniele Sterle SM3201594

// Stack memory and manipulation.
// Implements stack push/pop logic and stack operators like d, s, o, and D.

#include <stdlib.h>
#include "../include/stack.h"

tensor* create_stack(){
    tensor* stack = malloc(sizeof(tensor) * DEF_STACK_SIZE);
    
    if (stack == NULL) {
        return NULL;
    }

    return stack;
}

int push(tensor* stack, tensor element, int curr_stack_size, int idx_head){

    // Se array occupato al 100% raddoppiare size
    if (idx_head == curr_stack_size){
        // TODO: resize size x2
    }

    stack[idx_head] = element;
    idx_head++;
    return idx_head;
}

int pop(tensor* stack, int idx_head, tensor* out_element) {
    // Optional but recommended: Check for stack underflow
    if (idx_head <= 0) {
        printf("ERRORE: Lo stack è vuoto, impossibile fare pop.\n");
        return idx_head; // Return the unchanged index
    }

    // 1. Decrement the index to point to the actual top element
    idx_head--; 

    // 2. Dereference the pointer and assign the tensor to it
    *out_element = stack[idx_head]; 

    // 3. Return the updated index
    return idx_head;
}

tensor create_tensor(float* values, int rows, int columns) {
    tensor curr; // Allocated directly on the stack

    curr.values = values;
    curr.rows = rows;
    curr.columns = columns;

    return curr;
}

void free_stack(tensor* stack, int idx_head){

    if (stack == NULL) return;

    for (int i = 0; i < idx_head; i++){
        free(stack[i].values);
    }

    free(stack);
}