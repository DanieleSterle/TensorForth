// Daniele Sterle SM3201594

// Application entry point and parsing.
// Reads the source file passed via CLI, tokenizes the input, and routes tokens.

#ifndef include
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "stack.h"
#define BUFFER_SIZE 64
#endif

int main(int argc, char *argv[]) {

    if (argc < 2){
        printf("Non è stato passato nessun file\n");
        exit(1);
    }

    tensor* stack = create_stack();
    int s_size = DEF_STACK_SIZE;
    int idx_head = 0;

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
            free_stack(stack, idx_head);
            fclose(file);
            return 1;
        }
        
        if (isspace(curr)) {
            last_char = curr; // Aggiorna lo stato prima di saltare
            continue;
        }

        if (curr == '['){

            if (last_char  ==  ']') {
                printf("ERRORE: Formato non valido. I tensori devono essere separati da almeno uno spazio.\n");
                break;
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
                    
                    // CREARE util.c file con funzioni generiche
                    // creare funzione clear_all
                    free(values);
                    free_stack(stack, idx_head); // Clean up the stack
                    fclose(file);               // Close the file
                    return 1;
                }

                if (last_char == '[' && curr != ' ') {
                    printf("ERRORE: Formato non valido. Manca lo spazio dopo '['.\n");
                    free(values);
                    free_stack(stack, idx_head); // Clean up the stack
                    fclose(file);               // Close the file
                    return 1;
                }

                if (curr  ==  ' '){
                    if (b_idx > 0){
                        // If b_idx > 0, it means we just finished reading a float
                        buffer[b_idx] = '\0';
                        float curr_float = atof(buffer);

                        // --- INSERIRE IN VALUES + RESIZE ---

                        if (v_idx >= v_size){
                            v_size *= 2;
                            float* temp = (float*) realloc(values, sizeof(float) * v_size);
    
                            // 3. Check if realloc failed
                            if (temp  ==  NULL) {
                                printf("ERRORE: Memoria esaurita durante il ridimensionamento.\n");
                                free(values); // Safely free the old memory
                                free_stack(stack, idx_head); // Clean up the stack
                                fclose(file);               // Close the file
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
                    free(values);
                    free_stack(stack, idx_head); // Clean up the stack
                    fclose(file);               // Close the file
                    return 1;
                }

                // COSA SUCCEDE SE ho 2 -/. ?
                if (isdigit(curr)  ||  curr  ==  '.'  ||  curr  ==  '-'){
                    
                    if (curr == '-' && last_char != ' ') {
                        printf("ERRORE: Segno '-' in posizione non valida all'interno del numero.\n");
                        free(values);
                        free_stack(stack, idx_head); // Clean up the stack
                        fclose(file);               // Close the file
                        return 1;
                    }

                    if (b_idx < BUFFER_SIZE - 1){
                        buffer[b_idx++] = curr;
                    }else{
                        printf("ERRORE: Buffer overflow. Numero oltre %d caratteri non consentito.\n", BUFFER_SIZE);
                        free_stack(stack, idx_head); // Clean up the stack
                        fclose(file);               // Close the file
                        free(values); // Safely free the old memory
                        return 1;
                    }
                    last_char = curr;
                    continue;
                }
                

                printf("ERRORE: Carattere non valido '%c' (Codice ASCII: %d).\n", curr, curr);
                free(values);
                free_stack(stack, idx_head); // Clean up the stack
                fclose(file);               // Close the file
                return 1;

            }

            if (last_char != ' ' && last_char != '[') {
                printf("ERRORE: Formato non valido. Manca lo spazio prima di ']'.\n");
                free_stack(stack, idx_head); // Clean up the stack
                fclose(file);               // Close the file
                free(values);
                return 1;
            }

            last_char = ']';

            if (v_idx == 0) {
                printf("ERRORE: Tensore vuoto '[]' non consentito.\n");
                free(values);
                free_stack(stack, idx_head);
                fclose(file);
                return 1;
            }

            // SHRINKING MALLOC LOGIC
            //int old_v_size = v_size;
            v_size = v_idx;
            float* temp = (float*) realloc(values, sizeof(float) * v_size);
    
            // 3. Check if realloc failed
            if (temp  ==  NULL) {
                printf("ERRORE: Memoria esaurita durante il ridimensionamento.\n");
                free_stack(stack, idx_head); // Clean up the stack
                fclose(file);               // Close the file
                free(values); // Safely free the old memory
                return 1;     // Exit the program
            }
            // 4. It succeeded! Update the main pointer
            values = temp;

            //printf("ho shrinkato correttamente da %d a %d", old_v_size, v_size);

            idx_head = push(stack, create_tensor(values, 1, v_size), s_size, idx_head);
            
            //DEBUG
            printf("\nho fatto il push di:");
            printf("\n[");

            // Loop up to v_idx (the number of floats in this tensor), not idx_head
            for (int i = 0; i < v_idx; i++){
                printf(" %.2f ", values[i]);
            }

            printf("]\n");

        }

        if (curr == '"'){
            //TODO: gestione file 
        }

        // TODO swithc
        // CREARE util.c file con funzioni generiche
        // creare funzione clear_all
        switch (curr) {
        case '+':
            /* code */
            break;
        
        default:
            break;

        }
    }


    // Liberare tutta memoria
    fclose(file);
    free_stack(stack, idx_head); // Clean up the stack
    return 0;
}