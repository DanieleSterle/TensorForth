// Daniele Sterle SM3201594

#ifndef TENSORFORTH_TENSOR_H
#define TENSORFORTH_TENSOR_H

#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <sys/mman.h>
#include "utils.h"

// Numero massimo di dimensioni supportate per i tensori
#define MAX_DIM 2

// Tipi di dati supportati dal tensore
typedef enum {
    TYPE_NUMERIC,
    TYPE_NUMERIC_MMAP,
    TYPE_STRING
} tensor_type;

// Tipologie di forma strutturale del tensore
typedef enum {
    TENSOR_SHAPE_VECTOR,
    TENSOR_SHAPE_MATRIX
} tensor_shape;

// Struttura principale per la rappresentazione di un tensore
typedef struct {
    // Tag per identificare quale campo della union è attivo
    tensor_type type;
    
    union {                 
        // Dati per i tensori numerici (standard o in memoria)
        struct {            
            float* values;
            // shape[0] per 1D; shape[0] (righe), shape[1] (colonne) per 2D
            int32_t shape[MAX_DIM];
            // 1 per vettore, 2 per matrice
            int32_t ndim;
        };                  
        
        // Nome del file utilizzato per le operazioni di I/O
        char* filename;
    };
    
    // Contatore di riferimenti condiviso per la gestione della memoria
    int* ref_count;
} tensor;

// Struttura di supporto per la memorizzazione dei metadati su disco
typedef struct {
    int32_t shape[MAX_DIM];
    int32_t ndim;
    off_t data_offset;
} on_disk_tensor;

int shape_cmp(tensor* t1, tensor* t2);
int shape_cmp_matmul(tensor* t1, tensor* t2);
int is_boolean(tensor* t);
int is_matrix(tensor* t);
int is_vector(tensor* t);

int tensor_init_numeric(tensor* t, float* values, int* shape, int tensor_shape);
int tensor_init_string(tensor* t, char* string);
void free_tensor(tensor* t);

void tensor_print(tensor* t);

#endif