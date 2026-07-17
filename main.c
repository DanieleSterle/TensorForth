// Daniele Sterle SM3201594

// Application entry point and parsing.
// Reads the source file passed via CLI, tokenizes the input, and routes tokens.

#ifndef include
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "stack.h"
#include "utils.h"
#define BUFFER_SIZE 64
#endif

int main(int argc, char *argv[]) {

    if (argc < 2){
        printf("Non è stato passato nessun file\n");
        exit(1);
    }

    tensor* stack = NULL;
    int s_size = create_stack(&stack);
    int s_head = 0;

    if (s_size  ==  -1) {
        printf("ERRORE: Impossibile creare lo stack");
        return 1;
    }

    FILE* file = fopen(argv[1], "r"); 
    int curr;
    if (file == NULL) {
        printf("ERRORE: Impossibile aprire il file '%s'\n", argv[1]);
        return 1; // or exit(1);
    }

    int last_char = ' ';

    while ((curr = getc(file))  !=  EOF) {
        
        // Gestione numero fuori posto
        if (isdigit(curr)) {
            printf("ERRORE: Numero trovato al di fuori di un tensore.\n");
            free_all(file, stack, s_head, NULL);

            return 1;
        }
        
        if (isspace(curr)) {
            last_char = curr; // Aggiorna lo stato prima di saltare
            continue;
        }

        if (curr == '['){

            if (last_char  ==  ']') {
                printf("ERRORE: Formato non valido. I tensori devono essere separati da almeno uno spazio.\n");
                free_all(file, stack, s_head, NULL);
                return 1;
            }
            
            float* values = (float*) malloc(sizeof(float) * DEF_VALUES_SIZE);
            int v_idx = 0;
            int v_size = DEF_VALUES_SIZE;

            char buffer[BUFFER_SIZE];
            int b_idx = 0;
            
            last_char = '[';

            while((curr = getc(file))  !=  ']'){
                
                if (curr == EOF) {
                    printf("ERRORE: Fine file inaspettata prima di chiudere la parentesi ']'.\n");
                    free_all(file, stack, s_head, values);
                    return 1;
                }

                if (last_char == '[' && curr != ' ') {
                    printf("ERRORE: Formato non valido. Manca lo spazio dopo '['.\n");
                    free_all(file, stack, s_head, values);
                    return 1;
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
                            fprintf(stderr, "Error: String contains invalid trailing characters: '%s'\n", endptr);
                            free_all(file, stack, s_head, values);
                            return 1;
                        } 

                        // --- INSERIRE IN VALUES + RESIZE ---

                        if (v_idx >= v_size){
                            v_size *= 2;
                            float* temp = (float*) realloc(values, sizeof(float) * v_size);
    
                            // 3. Check if realloc failed
                            if (temp  ==  NULL) {
                                printf("ERRORE: Memoria esaurita durante il ridimensionamento.\n");
                                free_all(file, stack, s_head, values);
                                return 1;     // Exit the program
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
                    printf("ERRORE: Spazi bianchi non consentiti (newline/tab) dentro il tensore.\n");
                    free_all(file, stack, s_head, values);
                    return 1;
                }

                if (isdigit(curr)  ||  curr  ==  '.'  ||  curr  ==  '-'){
                    
                    if (curr == '-' && last_char != ' ') {
                        printf("ERRORE: Segno '-' in posizione non valida all'interno del numero.\n");
                        free_all(file, stack, s_head, values);
                        return 1;
                    }

                    if (b_idx < BUFFER_SIZE - 1){
                        buffer[b_idx++] = curr;
                    }else{
                        printf("ERRORE: Buffer overflow. Numero oltre %d caratteri non consentito.\n", BUFFER_SIZE);
                        free_all(file, stack, s_head, values);
                        return 1;
                    }
                    last_char = curr;
                    continue;
                }
                

                printf("ERRORE: Carattere non valido '%c' (Codice ASCII: %d).\n", curr, curr);
                free_all(file, stack, s_head, values);
                return 1;

            }

            if (last_char != ' ' && last_char != '[') {
                printf("ERRORE: Formato non valido. Manca lo spazio prima di ']'.\n");
                free_all(file, stack, s_head, values);
                return 1;
            }

            last_char = ']';

            if (v_idx == 0) {
                printf("ERRORE: Tensore vuoto '[]' non consentito.\n");
                free_all(file, stack, s_head, values);
                return 1;
            }

            // SHRINKING MALLOC LOGIC
            //int old_v_size = v_size;
            v_size = v_idx;
            float* temp = (float*) realloc(values, sizeof(float) * v_size);
    
            // 3. Check if realloc failed
            if (temp  ==  NULL) {
                printf("ERRORE: Memoria esaurita durante il ridimensionamento.\n");
                free_all(file, stack, s_head, values);
                return 1;     // Exit the program
            }
            // 4. It succeeded! Update the main pointer
            values = temp;

            //printf("ho shrinkato correttamente da %d a %d", old_v_size, v_size);

            tensor new_tensor;
            if (create_tensor(&new_tensor, values, 1, v_size)  ==  -1) {
                printf("ERRORE: Impossibile creare tensore.\n");
            }

            s_head = push(&stack, new_tensor, &s_size, s_head);
            
            if (s_head  ==  -1) {
                printf("ERRORE: Memoria esaurita durante il ridimensionamento.\n");
                free_all(file, stack, s_head, values);
                return 1;
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


        // SOLO CHIAMARE FUNZIONI + print errori, NO BUSINESS LOGIC
        switch (curr) {
        
        /* Operazioni aritmetiche */
        case '+':
            
            



            break;
        case '-':
            /* code */
            break;
        case '*':
            /* code */
            break;

        /* Operazioni di comparazione */
        case '<':
            /* code */
            break;
        case '>':
            /* code */
            break;
        case '=':
            /* code */
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