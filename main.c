// Daniele Sterle SM3201594

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "stack.h"
#include "utils.h"
#include "interpreter.h"

// Funzione Principale (Entry Point)
int main(int argc, char *argv[]) {
    
    // Inizializzazione del generatore di numeri casuali per l'operatore '?'
    srand(time(NULL));

    // Verifica della presenza del argomento obbligatorio (file sorgente)
    if (argc < 2){
        print_error(ERR_MISSING_ARGUMENT);
        return ERR_MISSING_ARGUMENT;
    }

    // Inizializzazione dello stack
    tensor* stack = NULL;
    int s_size = init_stack(&stack);
    int s_head = 0;

    if (s_size == ERR_OUT_OF_MEMORY) {
        print_error(ERR_OUT_OF_MEMORY);
        return ERR_OUT_OF_MEMORY;
    }

    // Passaggio del controllo al modulo interprete per l'esecuzione del file sorgente
    int status = interpret_file(argv[1], &stack, &s_size, &s_head);

    // Gestione di eventuali errori riscontrati durante l'esecuzione del programma
    if (status != ERR_SUCCESS) {
        print_error(status);
        free_stack(stack, s_head);
        return status;
    }

    // Rilascio delle risorse allocate per lo stack e terminazione con successo
    free_stack(stack, s_head);
    return status;
}