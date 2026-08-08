// Daniele Sterle SM3201594

#include "stack.h"

// Inizializza lo stack allocando la memoria iniziale in base alla capacità iniziale
int init_stack(tensor** stack) {
    int curr_stack_size = INITIAL_STACK_CAPACITY;

    if (curr_stack_size == 0) {
        curr_stack_size = 1;
    }

    *stack = malloc(sizeof(tensor) * curr_stack_size);
    
    if (*stack == NULL) {
        return ERR_OUT_OF_MEMORY;
    }

    return curr_stack_size;
}

// Inserisce un elemento nello stack, raddoppiandone la capacità se necessario
int push(tensor** stack, tensor t, int* curr_stack_size, int idx_head) {

    // Raddoppia la capacità dello stack quando l'array risulta completamente pieno
    if (idx_head == *curr_stack_size) {
        int new_size = *curr_stack_size * 2;

        tensor* temp = realloc(*stack, sizeof(tensor) * new_size);
        if (temp == NULL) {
            return ERR_OUT_OF_MEMORY;
        }

        *stack = temp;
        *curr_stack_size = new_size;
    }

    (*stack)[idx_head] = t;
    idx_head++;
    
    return idx_head;
}

// Rimuove e restituisce l'elemento in cima allo stack
int pop(tensor* stack, tensor* t, int idx_head) {

    if (idx_head <= 0) {
        return ERR_STACK_UNDERFLOW;
    }

    idx_head--; 

    *t = stack[idx_head];

    return idx_head;
}

// Dealloca interamente lo stack e tutti i tensori contenuti al suo interno
void free_stack(tensor* stack, int idx_head){

    if (stack == NULL) return;

    // Libera le risorse associate a ciascun tensore presente nello stack
    for (int i = 0; i < idx_head; i++){
        free_tensor(&stack[i]);
    }

    free(stack);
}

// Duplica l'elemento in cima allo stack incrementandone il ref_count
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

// Scambia la posizione dei primi due elementi in cima allo stack
int stack_swap(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2;

    // Estrae il primo elemento dalla cima
    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    // Estrae il secondo elemento dalla cima
    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    // Reinserisce il primo elemento estratto per invertire l'ordine
    int push_result = push(stack, t1, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&t1);
        free_tensor(&t2);
        return push_result;  
    }

    *s_head = push_result; 

    // Reinserisce il secondo elemento estratto completando lo scambio
    push_result = push(stack, t2, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&t2); 
        return push_result;  
    }

    *s_head = push_result; 
    return ERR_SUCCESS;
}

// Copia il secondo elemento dall'alto e lo posiziona in cima allo stack
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


// Rimuove e dealloca l'elemento attualmente in cima allo stack
int stack_drop(tensor** stack, int* s_head) {

    if (*s_head <= 0) {
        return ERR_STACK_UNDERFLOW;
    }

    tensor temp;
    *s_head = pop(*stack, &temp, *s_head);
    free_tensor(&temp);

    return ERR_SUCCESS;

}