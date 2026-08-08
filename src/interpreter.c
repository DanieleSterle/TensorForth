// Daniele Sterle SM3201594

#include "interpreter.h"

// Interpreta ed esegue il file sorgente passato come argomento leggendolo carattere per carattere
int interpret_file(const char* filename, tensor** stack, int* s_size, int* s_head) {
    
    // Apertura del file sorgente in modalità lettura
    FILE* file = fopen(filename, "r"); 
    if (file == NULL) {
        fprintf(stderr, "File: '%s'\n", filename); 
        return ERR_FILE_OPEN;
    }

    int curr;
    int last_char = ' ';
    int status = ERR_SUCCESS;

    // Scansione sequenziale del file fino alla fine
    while ((curr = getc(file)) != EOF) {
        // Ignora gli spazi bianchi e i caratteri di spaziatura
        if (isspace(curr)) {
            last_char = curr;
            continue;
        }

        // 1. Gestione del parsing dei tensori inline
        if (curr == '[') {
            status = parse_inline_tensor(file, stack, s_size, s_head, last_char);
            if (status != ERR_SUCCESS) break;
            last_char = ']';
            continue;
        }

        // 2. Gestione del parsing delle stringhe letterali
        if (curr == '"') {
            status = parse_string_literal(file, stack, s_size, s_head);
            if (status != ERR_SUCCESS) break;
            last_char = '"';
            continue;
        }

        // 3. Gestione e smistamento degli operatori del linguaggio
        status = execute_operator(curr, stack, s_size, s_head);
        if (status != ERR_SUCCESS) break;
        
        last_char = curr;
    }

    // Chiusura del file e restituzione dello stato di esecuzione
    fclose(file);
    return status;
}

// Analizza e costruisce un tensore inline racchiuso tra parentesi quadre
int parse_inline_tensor(FILE* file, tensor** stack, int* s_size, int* s_head, int last_char) {
    // Verifica sintattica: non è ammesso l'uso consecutivo senza spazi o caratteri non validi prima di '['
    if (last_char == ']' || last_char == '"') {
        return ERR_SYNTAX;
    }
    
    // Allocazione dinamica iniziale per il buffer dei valori float
    float* values = (float*) malloc(sizeof(float) * INITIAL_TENSOR_CAPACITY);
    if (values == NULL) {
        return ERR_OUT_OF_MEMORY;
    }

    int v_idx = 0;
    int v_size = INITIAL_TENSOR_CAPACITY;
    char buffer[MAX_FLOAT_STR_LEN];
    int b_idx = 0;
    int curr;
    
    last_char = '[';

    // Lettura dei singoli elementi numerici fino alla chiusura della parentesi ']'
    while ((curr = getc(file)) != ']') {
        
        if (curr == EOF) {
            free(values);
            return ERR_UNEXPECTED_EOF;
        }

        // Non sono ammessi numeri o caratteri attaccati direttamente all'apertura '['
        if (last_char == '[' && curr != ' ') {
            free(values);
            return ERR_SYNTAX;
        }

        
        if (curr == ' ') {

            // Se incontriamo uno spazio e abbiamo accumulato dei caratteri nel buffer,
            // significa che abbiamo appena completato la lettura di un numero float.
            // Il blocco procede quindi a convertire la stringa nel formato numerico,
            // a validarne la correttezza sintattica, a gestire l'eventuale ridimensionamento
            // dinamico dell'array dei valori e a memorizzarlo nell'elenco del tensore.
            if (b_idx > 0) {
                // Conversione della stringa accumulata nel buffer in un valore float
                buffer[b_idx] = '\0';
                char* endptr;

                // Converte la stringa in double (con cast a float) e aggiorna endptr per la validazione.
                // Nota: l'uso del terminatore '\0' e il reset di b_idx evitano qualsiasi sovrapposizione con i dati precedenti.
                float curr_float = (float) strtod(buffer, &endptr);

                // Controllo della correttezza del formato numerico
                if (*endptr != '\0') {
                    free(values);
                    return ERR_INVALID_NUMBER;
                } 

                // Ridimensionamento dinamico dell'array se la capacità massima corrente viene superata
                if (v_idx >= v_size) {
                    v_size *= 2;
                    float* temp = (float*) realloc(values, sizeof(float) * v_size);
                    if (temp == NULL) {
                        free(values);
                        return ERR_OUT_OF_MEMORY;
                    }
                    values = temp;
                }
                values[v_idx++] = curr_float;
                b_idx = 0;
            }
            last_char = curr;
            continue;
        }

        // I caratteri di nuova linea o tabulazione non sono consentiti all'interno dei tensori
        if (curr == '\n' || curr == '\t' || curr == '\r') {
            free(values);
            return ERR_SYNTAX;
        }

        // Acquisizione dei caratteri validi che compongono un numero float
        if (isdigit(curr) || curr == '.' || curr == '-') {
            // Il segno meno '-' è ammesso solo se preceduto da uno spazio
            if (curr == '-' && last_char != ' ') {
                free(values);
                return ERR_SYNTAX;
            }
            if (b_idx < MAX_FLOAT_STR_LEN - 1) {
                buffer[b_idx++] = curr;
            } else {
                // Superamento della lunghezza massima consentita per il numero
                free(values);
                return ERR_SYNTAX;
            }
            last_char = curr;
            continue;
        }
        
        free(values);
        return ERR_INVALID_CHAR;
    }

    // Verifica che la parentesi chiusa non segua direttamente un altro separatore non valido
    if (last_char != ' ' && last_char != '[') {
        free(values);
        return ERR_SYNTAX;
    }

    // Un tensore vuoto non è consentito dalla sintassi
    if (v_idx == 0) {
        free(values);
        return ERR_EMPTY_TENSOR;
    }

    // Ridimensionamento finale dell'array (shrink-to-fit) in base al numero effettivo di elementi
    v_size = v_idx;
    float* temp = (float*) realloc(values, sizeof(float) * v_size);
    if (temp == NULL) {
        free(values);
        return ERR_OUT_OF_MEMORY;
    }
    values = temp;

    // Configurazione della forma del tensore (inizialmente 1D / vettore)
    int shape[] = {v_size};

    // Inizializzazione del nuovo tensore e inserimento nello stack
    tensor new_tensor;
    int tensor_init_numeric_result = tensor_init_numeric(&new_tensor, values, shape, TENSOR_SHAPE_VECTOR);
    if (tensor_init_numeric_result != ERR_SUCCESS) {
        free(values);
        return tensor_init_numeric_result;
    }

    int push_result = push(stack, new_tensor, s_size, *s_head);
    if (push_result < 0) {
        free(values);
        return push_result;
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

// Analizza e acquisisce una stringa letterale racchiusa tra doppi apici (es. per i percorsi dei file)
int parse_string_literal(FILE* file, tensor** stack, int* s_size, int* s_head) {
    
    char buffer_file[MAX_FILENAME_LEN];
    int b_idx = 0;
    int curr;
    
    // Lettura dei caratteri fino al raggiungimento del doppio apice di chiusura
    while ((curr = getc(file)) != '"') {
        
        if (curr == EOF) {
            return ERR_UNEXPECTED_EOF;
        }

        if (b_idx >= MAX_FILENAME_LEN) { 
            return ERR_BUFFER_OVERFLOW;
        }

        buffer_file[b_idx] = curr;
        b_idx++;
    }
    
    buffer_file[b_idx] = '\0';
    
    // Inizializzazione del tensore stringa e inserimento nello stack
    tensor new_tensor;
    int tensor_init_string_result = tensor_init_string(&new_tensor, buffer_file);
    if (tensor_init_string_result != ERR_SUCCESS) {
        return tensor_init_string_result;
    }

    int push_result = push(stack, new_tensor, s_size, *s_head);
    if (push_result < 0) {
        return push_result; // push_result contains the error code
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

// Smista l'operatore letto dal file sorgente verso la specifica funzione di gestione
int execute_operator(int op_char, tensor** stack, int* s_size, int* s_head) {
    switch (op_char) {
        
        /* Operazioni Aritmetiche */
        case '+':
            return handle_add_op(stack, s_size, s_head);
            break;
        case '-':
            return handle_subtract_op(stack, s_size, s_head);
            break;
        case '*':
            return handle_multiply_op(stack, s_size, s_head);
            break;

        /* Operazioni di Comparazione */
        case '<':
            return handle_lt_op(stack, s_size, s_head);
            break;
        case '>':
            return handle_gt_op(stack, s_size, s_head);
            break;
        case '=':
            return handle_eq_op(stack, s_size, s_head);
            break;

        /* Operazioni Logiche */
        case '&':
            return handle_and_op(stack, s_size, s_head);
            break;
        case '|':
            return handle_or_op(stack, s_size, s_head);
            break;
        case '!':
            return handle_not_op(stack, s_size, s_head);
            break;

        /* Operazioni di Selezione */
        case '$':
            return handle_select_op(stack, s_size, s_head);
            break;

        /* Operazioni Specifiche per Tensori */
        case '@':
            return handle_matmul_op(stack, s_size, s_head);
            break;
        case '.':
            return handle_dot_op(stack, s_size, s_head);
            break;
        case 'c':
            return handle_conv2d_op(stack, s_size, s_head);
            break;

        /* Operazioni sulla Forma dei Tensori */
        case 'r':
            return handle_reshape_op(stack, s_size, s_head);
            break;
        case '_':
            return handle_ravel_op(stack, s_size, s_head);
            break;
        case '#':
            return handle_get_shape_op(stack, s_size, s_head);
            break;

        /* Operazioni di Generazione di Numeri Casuali */
        case '?':
            return handle_random_op(stack, s_size, s_head);
            break;

        /* Operazioni Elemento per Elemento (Unarie/Binarie) */
        case 'R':
            return handle_relu_op(stack, s_size, s_head);
            break;
        case 'm':
            return handle_min_op(stack, s_size, s_head);
            break;
        case 'M':
            return handle_max_op(stack, s_size, s_head);
            break;

        /* Operazioni di Riduzione */
        case 'S':
            return handle_sum_reduce_op(stack, s_size, s_head);
            break;

        /* Operazioni di Filling dei Tensori */
        case 'f':
            return handle_fill_op(stack, s_size, s_head);
            break;

        /* Operazioni di Utilità */
        case 'p':
            return handle_print_op(stack, s_head);
            break;

        /* Operazioni di Manipolazione dello Stack (Stile Forth) */
        case 'd':
            return stack_dup(stack, s_size, s_head);
            break;
        case 's':
            return stack_swap(stack, s_size, s_head);
            break;
        case 'o':
            return stack_over(stack, s_size, s_head);
            break;
        case 'D':
            return stack_drop(stack, s_head);
            break;

        /* Operazioni di I/O (File e Immagini PGM) */
        case '(':
            return handle_read_pgm_op(stack, s_size, s_head);
            break;
        case ')':
            return handle_write_pgm_op(stack, s_head);
            break;
        case '{':
            return handle_read_mmap_op(stack, s_size, s_head);
            break;
        case '}':
            return handle_write_bin_op(stack, s_head);
            break;
        
        default:
            return ERR_SYNTAX;
            break;
    }
}