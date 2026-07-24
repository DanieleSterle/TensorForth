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

int shape_cmp_matmul(tensor* t1, tensor* t2) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)) {
        return ERR_NULL_PTR; 
    }

    if (t1->columns  ==  t2->rows){
        return ERR_SUCCESS;
    }

    return ERR_MATMUL_DIM_MISMATCH;

}

int shape_cmp_dot(tensor* t1, tensor* t2) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)) {
        return ERR_NULL_PTR; 
    }

    int len_t1 = t1->rows * t1->columns;
    int len_t2 = t2->rows * t2->columns;

    if (len_t1  ==  len_t2) {
        return ERR_SUCCESS;
    }

    return ERR_DOT_DIM_MISMATCH; 
}

int is_boolean(tensor* t) {

    if (t  ==  NULL) return ERR_NULL_PTR; 

    /*
    OPTIMIZE WITH OPENMP
    for (int i = 0; i < res->rows; i++){
        for (int j = 0; j < res->columns; j++){
            result->values[i] = (t1->values[i] == t2->values[i]);
        }
    }
    */

    int s_values = t->rows * t->columns;
    for (int i = 0; i < s_values; i++) {
       if ((t->values[i]  !=  0.0)  &&  (t->values[i]  !=  1.0)) return ERR_NOT_BOOLEAN;
    }

    return ERR_SUCCESS;

}

int is_matrix(tensor* t) {

    if (t  ==  NULL) return ERR_NULL_PTR;

    if ((t->rows  !=  1)  &&  (t->columns  !=  1)){
        return ERR_SUCCESS;
    }

    return ERR_NOT_MATRIX;

}

int is_vector(tensor* t) {

    if (t  ==  NULL) return ERR_NULL_PTR;

    if ((t->rows == 1) || (t->columns == 1)) {
        return ERR_SUCCESS;
    }

    return ERR_NOT_VECTOR;

}

void tensor_print(tensor* t) {
    
    if (t == NULL) {
        return;
    }

    // shape: rows columns
    printf("Tensor(shape=[");
    printf("%d %d", t->rows, t->columns);
    
    printf("], data=[");

    // Print all data values separated by a space
    int total_elements = t->rows * t->columns;
    for (int i = 0; i < total_elements; i++) {
        printf("%.2f", t->values[i]);
        if (i < total_elements - 1) {
            printf(" ");
        }
    }
    
    printf("])\n");
}

int tensor_generate_random(tensor* s, tensor* result) {
    if ((s  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int vector_result = is_vector(s);
    if (vector_result  !=  ERR_SUCCESS) return vector_result;

    int s_len = s->rows * s->columns;

    if ((s_len < 1)  ||  (s_len > 2)) {
        return ERR_SHAPE_MISMATCH; 
    }

    int rows;
    int cols;

    if (s_len == 2) {
        // shape is 2D Matrix
        rows = (int)s->values[0];
        cols = (int)s->values[1];
    } else {
        // shape is 1D Vector 
        rows = 1;
        cols = (int)s->values[0];
    }

    if ((rows <= 0)  ||  (cols <= 0)) {
        return ERR_SHAPE_MISMATCH;
    }

    result->columns = cols;
    result->rows = rows;

    int size = cols * rows;
    result->values = (float*) malloc(sizeof(float) * size);

    if (result->values  ==  NULL) {
        return ERR_OUT_OF_MEMORY;
    }

    // NON OTTIMIZZARE
    for (int i = 0; i < size; i++){
        result->values[i] = (float)rand() / (float)RAND_MAX;
    }

    return ERR_SUCCESS;

}

void print_error(int error_code) {
    switch (error_code) {
        case ERR_SUCCESS:
            break;
        case ERR_NULL_PTR:
            fprintf(stderr, "ERRORE: Rilevato puntatore nullo.\n");
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
        case ERR_SHAPE_MISMATCH:
            fprintf(stderr, "ERRORE: Le dimensioni dei tensori non corrispondono per questa operazione.\n");
            break;
        case ERR_NOT_BOOLEAN:
            fprintf(stderr, "ERRORE LOGICO: I tensori per questa operazione accettano esclusivamente valori booleani (0.0 o 1.0).\n");
            break;
        case ERR_NOT_MATRIX:
            fprintf(stderr, "ERRORE DI DIMENSIONE: L'operazione richiede una matrice (tensore 2D), ma è stato fornito un tensore 1D.\n");
            break;
        case ERR_DOT_DIM_MISMATCH:
            fprintf(stderr, "ERRORE DI DIMENSIONE: I vettori per il prodotto interno (dot product) devono avere lo stesso numero totale di elementi.\n");
            break;
        case ERR_MATMUL_DIM_MISMATCH:
            fprintf(stderr, "ERRORE DI DIMENSIONE: Tensori non compatibili per il prodotto. Le colonne del primo tensore devono coincidere con le righe del secondo.\n");
            break;
        case ERR_MISSING_ARGUMENT:
            fprintf(stderr, "ERRORE: Non è stato passato nessun file in input.\n");
            break;
        case ERR_FILE_OPEN:
            fprintf(stderr, "ERRORE: Impossibile aprire il file specificato.\n");
            break;
        case ERR_UNEXPECTED_EOF:
            fprintf(stderr, "ERRORE: Fine file inaspettata prima della chiusura ']'.\n");
            break;
        case ERR_SYNTAX:
            fprintf(stderr, "ERRORE DI SINTASSI: Formato non valido, spazio mancante o numero fuori posto.\n");
            break;
        case ERR_INVALID_CHAR:
            fprintf(stderr, "ERRORE: Trovato carattere ASCII non valido.\n");
            break;
        case ERR_BUFFER_OVERFLOW:
            fprintf(stderr, "ERRORE: Buffer overflow. Il numero supera i caratteri consentiti.\n");
            break;
        case ERR_EMPTY_TENSOR:
            fprintf(stderr, "ERRORE DI SINTASSI: Tensore vuoto '[]' non consentito.\n");
            break;
        case ERR_INVALID_NUMBER:
            fprintf(stderr, "ERRORE DI SINTASSI: Formato numerico non valido (es. segno '-' errato o multipli punti).\n");
            break;
        default:
            fprintf(stderr, "ERRORE: Si è verificato un errore sconosciuto (Codice: %d).\n", error_code);
            break;
    }
}