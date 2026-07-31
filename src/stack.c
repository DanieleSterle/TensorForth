// Daniele Sterle SM3201594

// Stack memory and manipulation.
// Implements stack push/pop logic and stack operators like d, s, o, and D.

#include <stdlib.h>
#include "stack.h"
#include "utils.h"

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
        return ERR_OUT_OF_MEMORY;
    }

    // Return the initial capacity
    return curr_stack_size;
}

int push(tensor** stack, tensor t, int* curr_stack_size, int idx_head) {

    // Se array occupato al 100% raddoppiare size
    if (idx_head == *curr_stack_size) {
        int new_size = *curr_stack_size * 2;

        tensor* temp = realloc(*stack, sizeof(tensor) * new_size);
        if (temp == NULL) {
            return ERR_OUT_OF_MEMORY; // Allocation failed
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

    if (idx_head <= 0) {
        return ERR_STACK_UNDERFLOW;
    }

    idx_head--; 

    *t = stack[idx_head];

    return idx_head;
}

// CAMBIARE NOME?
int create_numeric_tensor(tensor* t, float* values, int rows, int columns) {

    if (t  ==  NULL) {
        return ERR_NULL_PTR; 
    }
    
    if ((rows <= 0)  ||  (columns <= 0)) {
        return ERR_SHAPE_MISMATCH; 
    }

    t->type = TYPE_NUMERIC;
    t->rows = rows;
    t->columns = columns;

    int newly_allocated = 0;

    if (values  ==  NULL) {
        int s_values = rows * columns;
        float* temp = (float*) malloc(sizeof(float) * s_values);

        if (temp  ==  NULL) return ERR_OUT_OF_MEMORY;

        t->values = temp;
        newly_allocated = 1;

    } else {
        t->values = values;
    }

    t->ref_count = (int*) malloc(sizeof(int));
    
    if (t->ref_count  ==  NULL) {
        if (newly_allocated  ==  1) {
            free(t->values); 
        }
        return ERR_OUT_OF_MEMORY;
    }

    (*t->ref_count) = 1;

    return ERR_SUCCESS;
}

// CAMBIARE NOME?
int create_string_tensor(tensor* t, char* string) {
    if ((t  ==  NULL)  ||  (string  ==  NULL)) {
        return ERR_NULL_PTR; 
    }
    
    t->type = TYPE_STRING;
    t->filename = string;

    t->ref_count = (int*) malloc(sizeof(int));
    if (t->ref_count  ==  NULL) {
        return ERR_OUT_OF_MEMORY;
    }

    (*t->ref_count) = 1;

    return ERR_SUCCESS;
}

void free_stack(tensor* stack, int idx_head){

    if (stack == NULL) return;

    for (int i = 0; i < idx_head; i++){
        free_tensor(&stack[i]);
    }

    free(stack);
}

int stack_dup(tensor** stack, int* s_size, int* s_head) {

    if (*s_head <= 0) {
        return ERR_STACK_UNDERFLOW;
    }

    tensor temp = (*stack)[*s_head - 1];
    (*temp.ref_count)++;

    int push_result = push(stack, temp, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&temp); 
        return push_result;  
    }

    *s_head = push_result; 
    return ERR_SUCCESS;

}

int stack_swap(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int push_result = push(stack, t1, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&t1);
        free_tensor(&t2);
        return push_result;  
    }

    *s_head = push_result; 

    push_result = push(stack, t2, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&t2); 
        return push_result;  
    }

    *s_head = push_result; 
    return ERR_SUCCESS;
}

int stack_over(tensor** stack, int* s_size, int* s_head) {

    if (*s_head <= 1) {
        return ERR_STACK_UNDERFLOW;
    }

    tensor temp = (*stack)[*s_head - 2];
    (*temp.ref_count)++;

    int push_result = push(stack, temp, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&temp); 
        return push_result;  
    }

    *s_head = push_result; 
    return ERR_SUCCESS;
}


int stack_drop(tensor** stack, int* s_head) {

    if (*s_head <= 0) {
        return ERR_STACK_UNDERFLOW;
    }

    tensor temp;
    *s_head = pop(*stack, &temp, *s_head);
    free_tensor(&temp);

    return ERR_SUCCESS;

}