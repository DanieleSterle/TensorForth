// Daniele Sterle SM3201594

#include "tensor.h"

// Confronta le dimensioni di due tensori verificandone la compatibilità esatta
int shape_cmp(tensor* t1, tensor* t2) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)) {
        return ERR_NULL_PTR; 
    }

    if (t1->ndim == t2->ndim) {
        if (t1->ndim == 1) {
            if (t1->shape[0] == t2->shape[0]) {
                return ERR_SUCCESS;
            }
        } else {
            if ((t1->shape[0] == t2->shape[0]) && (t1->shape[1] == t2->shape[1])) {
                return ERR_SUCCESS;
            }
        }
    }

    return ERR_SHAPE_MISMATCH;

}

// Verifica la compatibilità dimensionale per la moltiplicazione di matrici (colonne della prima == righe della seconda)
int shape_cmp_matmul(tensor* t1, tensor* t2) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)) {
        return ERR_NULL_PTR; 
    }

    if (t1->shape[1]  ==  t2->shape[0]){
        return ERR_SUCCESS;
    }

    return ERR_MATMUL_DIM_MISMATCH;

}

// Controlla che tutti i valori interni del tensore siano rigorosamente booleani (0.0 o 1.0)
int is_boolean(tensor* t) {

    if (t  ==  NULL) return ERR_NULL_PTR; 

    int s_values;
    if (t->ndim == 1) {
        s_values = t->shape[0];
    } else {
        s_values = t->shape[0] * t->shape[1];
    }
    
    int is_valid = 1;

    #pragma omp parallel for reduction(&&:is_valid)
    for (int i = 0; i < s_values; i++) {
        if ((t->values[i]  !=  0.0)  &&  (t->values[i]  !=  1.0)) {
            is_valid = 0;
        }
    }

    if (!is_valid) {
        return ERR_NOT_BOOLEAN;
    }

    return ERR_SUCCESS;

}

// Valida se il tensore corrisponde a una matrice 2D
int is_matrix(tensor* t) {
    
    if (t  ==  NULL) return ERR_NULL_PTR;

    if (t->ndim == 2) {
        return ERR_SUCCESS;
    }

    return ERR_NOT_MATRIX;

}

// Valida se il tensore corrisponde a un vettore 1D
int is_vector(tensor* t) {

    if (t  ==  NULL) return ERR_NULL_PTR;

    if (t->ndim == 1) {
        return ERR_SUCCESS;
    }

    return ERR_NOT_VECTOR;

}

// Inizializza un tensore di tipo numerico allocandone eventualmente i valori e il contatore dei riferimenti
int tensor_init_numeric(tensor* t, float* values, int* shape, int tensor_shape) {

    if (t  ==  NULL) {
        return ERR_NULL_PTR; 
    }
    
    if (shape == NULL) {
        return ERR_NULL_PTR;
    }
    
    t->type = TYPE_NUMERIC;
    int s_values = 0;
    
    // Controlla le forme in base al tipo di tensore per evitare letture fuori dai limiti
    if (tensor_shape == TENSOR_SHAPE_VECTOR) {
        if (shape[0] <= 0) return ERR_SHAPE_MISMATCH;
        
        t->ndim = 1;
        t->shape[0] = shape[0];
        t->shape[1] = 0;
        s_values = shape[0];
        
    } else if (tensor_shape == TENSOR_SHAPE_MATRIX) {
        if (shape[0] <= 0 || shape[1] <= 0) return ERR_SHAPE_MISMATCH;
        
        t->ndim = 2;
        t->shape[0] = shape[0];
        t->shape[1] = shape[1];
        s_values = shape[0] * shape[1];
        
    } else {
        return ERR_SHAPE_MISMATCH; 
    }

    int newly_allocated = 0;

    // Alloca dinamicamente la memoria per i valori se non sono stati forniti in input
    if (values  ==  NULL) {
        float* temp = (float*) malloc(sizeof(float) * s_values);

        if (temp  ==  NULL) return ERR_OUT_OF_MEMORY;

        t->values = temp;
        newly_allocated = 1;

    } else {
        t->values = values;
    }

    // Inizializza il ref_count a 1
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

// Inizializza un tensore di tipo stringa memorizzando il nome del file associato
int tensor_init_string(tensor* t, char* string) {
    if ((t  ==  NULL)  ||  (string  ==  NULL)) {
        return ERR_NULL_PTR; 
    }
    
    t->type = TYPE_STRING;

    // Alloca la memoria per la stringa aggiungendo lo spazio per il terminatore null ('\0')
    int str_len = strlen(string);
    t->filename = (char*) malloc(str_len + 1);
    
    if (t->filename == NULL) {
        return ERR_OUT_OF_MEMORY;
    }
    
    // Copia il contenuto della stringa nel buffer allocato
    strcpy(t->filename, string);

    // Alloca e inizializza il contatore dei riferimenti a 1
    t->ref_count = (int*) malloc(sizeof(int));
    if (t->ref_count  ==  NULL) {
        return ERR_OUT_OF_MEMORY;
    }

    (*t->ref_count) = 1;

    return ERR_SUCCESS;
}

void free_tensor(tensor* t) {
    if ((t == NULL) || (t->ref_count == NULL)) {
        return;
    }

    // Decrementa il contatore dei riferimenti condivisi
    (*t->ref_count)--;

    // Libera effettivamente le risorse solo quando nessun altro fa riferimento al tensore
    if ((*t->ref_count) == 0) {
        
        // Dealloca il buffer dei valori se il tensore è di tipo numerico standard
        if (t->type == TYPE_NUMERIC) {
            if (t->values != NULL) {
                free(t->values);
            }
        } 
        
        // Rilascia la proiezione in memoria (mmap) se il tensore proviene da file su disco
        if (t->type == TYPE_NUMERIC_MMAP) {
            if (t->values != NULL) {
                
                // Calcola il numero totale di elementi
                int total_elements;
                if (t->ndim == 1) {
                    total_elements = t->shape[0];
                } else {
                    total_elements = t->shape[0] * t->shape[1];
                }

                // Calcola la dimensione totale esatta del file in memoria
                size_t total_size = sizeof(on_disk_tensor) + 
                                   (total_elements * sizeof(float));

                // Torna indietro rispetto a t->values per risalire al puntatore mmap originale
                void* original_mmap_ptr = (void*)((char*)t->values - sizeof(on_disk_tensor));

                munmap(original_mmap_ptr, total_size);
            }
        } 
        
        // Libera la stringa allocata se il tensore contiene un percorso o un nome file
        if (t->type == TYPE_STRING) {
            if (t->filename != NULL) {
                free(t->filename);
            }
        }

        // Infine dealloca la memoria del contatore dei riferimenti
        free(t->ref_count);
    }
}

// Stampa a schermo la forma e i dati numerici del tensore
void tensor_print(tensor* t) {
    
    if (t == NULL) {
        return;
    }

    if ((t->type  !=  TYPE_NUMERIC)  &&  (t->type  !=  TYPE_NUMERIC_MMAP)) return;

    printf("Tensor(shape=[");
    if (t->ndim == 1) {
        printf("%d", t->shape[0]);
    } else {
        printf("%d %d", t->shape[0], t->shape[1]);
    }
    
    printf("], data=[");

    int total_elements;
    if (t->ndim == 1) {
        total_elements = t->shape[0];
    } else {
        total_elements = t->shape[0] * t->shape[1];
    }
    
    for (int i = 0; i < total_elements; i++) {
        printf("%.2f", t->values[i]);
        if (i < total_elements - 1) {
            printf(" ");
        }
    }
    
    printf("])\n");
}