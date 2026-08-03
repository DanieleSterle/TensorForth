// Daniele Sterle SM3201594

// Helpers, printing, and randomness.
// Implements the print function p, random tensor generation ?, and error handling.

#include "utils.h"

void print_error(int error_code) {
    switch (error_code) {
        case ERR_SUCCESS:
            break;
            
        // --- Memory & Stack ---
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
            
        // --- Math & Logic Operations ---
        case ERR_SHAPE_MISMATCH:
            fprintf(stderr, "ERRORE DI DIMENSIONE: Le dimensioni dei tensori non corrispondono per questa operazione.\n");
            break;
        case ERR_NOT_BOOLEAN:
            fprintf(stderr, "ERRORE LOGICO: I tensori per questa operazione accettano esclusivamente valori booleani (0.0 o 1.0).\n");
            break;
        case ERR_NOT_MATRIX:
            fprintf(stderr, "ERRORE DI DIMENSIONE: L'operazione richiede una matrice (tensore 2D), ma è stato fornito un tensore 1D.\n");
            break;
        case ERR_NOT_VECTOR:
            fprintf(stderr, "ERRORE DI DIMENSIONE: L'operazione richiede un vettore (tensore 1D).\n");
            break;
        case ERR_DOT_DIM_MISMATCH:
            fprintf(stderr, "ERRORE DI DIMENSIONE: I vettori per il prodotto interno devono avere lo stesso numero totale di elementi.\n");
            break;
        case ERR_MATMUL_DIM_MISMATCH:
            fprintf(stderr, "ERRORE DI DIMENSIONE: Tensori non compatibili per il prodotto. Le colonne del primo devono coincidere con le righe del secondo.\n");
            break;
        case ERR_EXPECTED_NUMERIC:
            fprintf(stderr, "ERRORE DI TIPO: L'operazione richiede un tensore numerico, ma è stata trovata una stringa.\n");
            break;
        case ERR_EXPECTED_STRING:
            fprintf(stderr, "ERRORE DI TIPO: L'operazione richiede una stringa (nome file), ma è stato trovato un tensore numerico.\n");
            break;

        // --- I/O and Files ---
        case ERR_MISSING_ARGUMENT:
            fprintf(stderr, "ERRORE: Non è stato passato nessun file in input.\n");
            break;
        case ERR_FILE_OPEN:
            fprintf(stderr, "ERRORE I/O: Impossibile aprire o creare il file specificato.\n");
            break;
        case ERR_UNEXPECTED_EOF:
            fprintf(stderr, "ERRORE: Fine file inaspettata prima della chiusura ']'.\n");
            break;
        case ERR_FILE_READ:
            fprintf(stderr, "ERRORE I/O: Impossibile leggere i dati dal file.\n");
            break;
        case ERR_FILE_WRITE:
            fprintf(stderr, "ERRORE I/O: Impossibile scrivere i dati nel file.\n");
            break;
        case ERR_INVALID_PGM:
            fprintf(stderr, "ERRORE DI FORMATO: Il file non è un'immagine PGM binaria valida (P5) o l'intestazione è corrotta.\n");
            break;
        case ERR_FILE_STAT:
            fprintf(stderr, "ERRORE DI SISTEMA: Impossibile recuperare le informazioni del file (fileno/fstat falliti).\n");
            break;
        case ERR_MMAP_FAILED:
            fprintf(stderr, "ERRORE DI MEMORIA: Mappatura del file in memoria (mmap) fallita.\n");
            break;
        case ERR_IO:
            fprintf(stderr, "ERRORE I/O: Si è verificato un errore generico di input/output.\n");
            break;

        // --- Syntax and Parsing ---
        case ERR_SYNTAX:
            fprintf(stderr, "ERRORE DI SINTASSI: Formato non valido, spazio mancante o numero fuori posto.\n");
            break;
        case ERR_INVALID_CHAR:
            fprintf(stderr, "ERRORE: Trovato carattere ASCII non valido nel file sorgente.\n");
            break;
        case ERR_BUFFER_OVERFLOW:
            fprintf(stderr, "ERRORE: Buffer overflow durante il parsing. Il numero supera i caratteri consentiti.\n");
            break;
        case ERR_EMPTY_TENSOR:
            fprintf(stderr, "ERRORE DI SINTASSI: Tensore vuoto '[]' non consentito.\n");
            break;
        case ERR_INVALID_NUMBER:
            fprintf(stderr, "ERRORE DI SINTASSI: Formato numerico non valido (es. segno '-' errato o punti multipli).\n");
            break;
            
        default:
            fprintf(stderr, "ERRORE: Si è verificato un errore sconosciuto (Codice: %d).\n", error_code);
            break;
    }
}