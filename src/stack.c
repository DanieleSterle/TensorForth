// Daniele Sterle SM3201594

// Stack memory and manipulation.
// Implements stack push/pop logic and stack operators like d, s, o, and D.

#include <stdlib.h>
#include "../include/stack.h"

int create_stack(tensor** stack) {
    int curr_stack_size = DEF_STACK_SIZE;

    // Se DEF_STACK_SIZE == 0, mettere 1
    if (curr_stack_size == 0) {
        curr_stack_size = 1;
    }

    // Allocate memory and assign it to the dereferenced pointer
    *stack = malloc(sizeof(tensor) * curr_stack_size);
    
    // Check if allocation failed
    if (*stack == NULL) {
        return -1;
    }

    // Return the initial capacity
    return curr_stack_size;
}

int push(tensor** stack, tensor t, int* curr_stack_size, int idx_head) {

    // Se array occupato al 100% raddoppiare size
    if (idx_head == *curr_stack_size) {
        // Handle the case where initial size is 0
        int new_size = *curr_stack_size * 2;

        tensor* temp = realloc(*stack, sizeof(tensor) * new_size);
        if (temp == NULL) {
            return -1; // Allocation failed
        }

        *stack = temp;              // Update the caller's pointer
        *curr_stack_size = new_size; // Update the caller's size variable
    }

    // Insert the t using the dereferenced stack pointer
    (*stack)[idx_head] = t;
    idx_head++;
    
    return idx_head;
}

int pop(tensor* stack, tensor* t, int idx_head) {
    // Optional but recommended: Check for stack underflow
    if (idx_head <= 0) {
        //printf("ERRORE: Lo stack è vuoto, impossibile fare pop.\n");
        return -1;
    }

    // 1. Decrement the index to point to the actual top t
    idx_head--; 

    // 2. Dereference the pointer and assign the tensor to it
    *t = stack[idx_head]; 

    // 3. Return the updated index
    return idx_head;
}

int create_tensor(tensor* t, float* values, int rows, int columns) {
    // Error handling: Check for NULL pointers or invalid dimensions
    if (t == NULL || values == NULL) {
        return -1; 
    }
    
    if (rows <= 0 || columns <= 0) {
        return -1; 
    }

    t->values = values;
    t->rows = rows;
    t->columns = columns;

    return 0;
}

void free_stack(tensor* stack, int idx_head){

    if (stack == NULL) return;

    for (int i = 0; i < idx_head; i++){
        free(stack[i].values);
    }

    free(stack);
}