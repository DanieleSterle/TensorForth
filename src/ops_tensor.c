// Daniele Sterle SM3201594

// In quasi tutte le funzioni di questo modulo viene eseguito un blocco iniziale standard 
// di validazione e preparazione, strutturato come segue:
// 1. Controllo dei puntatori NULL per evitare crash di dereferenziazione.
// 2. Verifica dei dati numerici tramite macro (es. ASSERT_NUMERIC).
// 3. Controllo di compatibilità della forma e del tipo (es. is_vector, is_matrix, shape_cmp).
// 4. Inizializzazione o calcolo delle nuove dimensioni del tensore risultato tramite tensor_init_numeric.
// Tale sequenza di controlli preliminari viene eseguita all'inizio di ogni funzione per garantire 
// la robustezza e la sicurezza dei dati prima di procedere con le elaborazioni numeriche o parallele.
// Per evitare ripetizioni inutili all'interno del codice, i controlli iniziali non sono commentati 
// singolarmente in ogni funzione.

#include "ops_tensor.h"

int tensor_matmul(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int matrix_result;
    matrix_result = is_matrix(t1);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    matrix_result = is_matrix(t2);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    int matmul_shape_result = shape_cmp_matmul(t1, t2);
    if (matmul_shape_result != ERR_SUCCESS) return matmul_shape_result;

    int shape[] = {t1->shape[0], t2->shape[1]};

    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    // Esegue il calcolo della moltiplicazione tra matrici utilizzando l'ordine i-k-j.
    // Inizializza prima tutti gli elementi del risultato a 0.0f (passaggio richiesto da questo 
    // specifico ordine dei cicli). Successivamente, calcola il prodotto sfruttando il loop più 
    // interno su 'j', che garantisce un accesso alla memoria sequenziale ottimale. 
    // Nota: Non è possibile usare collapse(2) poiché le variabili 'i' e 'j' non sono adiacenti.
    int res_elements = result->shape[0] * result->shape[1];
    #pragma omp parallel for
    for (int i = 0; i < res_elements; i++) {
        result->values[i] = 0.0f;
    }

    #pragma omp parallel for
    for (int i = 0; i < result->shape[0]; i++){
        for (int k = 0; k < t1->shape[1]; k++) {
            
            float t1_val = t1->values[i * t1->shape[1] + k];
            
            for (int j = 0; j < result->shape[1]; j++){
                result->values[i * result->shape[1] + j] += t1_val * t2->values[k * t2->shape[1] + j];
            }
        }
    }

    return ERR_SUCCESS;
}

int tensor_dot(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int vector_result;
    vector_result = is_vector(t1);
    if (vector_result != ERR_SUCCESS) return vector_result;

    vector_result = is_vector(t2);
    if (vector_result != ERR_SUCCESS) return vector_result;

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int shape[] = {1};
    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    // Esegue il calcolo del prodotto scalare in parallelo utilizzando una riduzione per la somma.
    float sum = 0.0f;
    int vector_length = t1->shape[0]; 
    
    #pragma omp parallel for reduction(+:sum)
    for (int k = 0; k < vector_length; k++) {
        sum += t1->values[k] * t2->values[k];
    }
    
    result->values[0] = sum;

    return ERR_SUCCESS;
}

int tensor_conv2d(tensor* t, tensor* k, tensor* result) {

    if ((t  ==  NULL)  ||  (k  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t);
    ASSERT_NUMERIC(k);

    int matrix_result;
    matrix_result = is_matrix(t);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    matrix_result = is_matrix(k);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    int shape[] = {t->shape[0], t->shape[1]};
    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }
    
    // Calcola gli offset centrali del kernel per gestire il posizionamento e il padding.
    int offset_row = k->shape[0] / 2;
    int offset_col = k->shape[1] / 2;

    // Esegue la convoluzione 2D in parallelo su due dimensioni (righe e colonne del risultato).
    // Per ogni pixel (i, j), determina i limiti validi di scansione del kernel per evitare 
    // l'uso di istruzioni condizionali 'if' nei cicli interni.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < result->shape[0]; i++) {
        for (int j = 0; j < result->shape[1]; j++) {
            
            float sum = 0.0f;

            // Determina i limiti di scorrimento sul kernel per escludere i bordi fuori dall'immagine.
            int start_x = (offset_row > i) ? (offset_row - i) : 0;
            int max_x   = t->shape[0] - i + offset_row;
            int end_x   = (k->shape[0] < max_x) ? k->shape[0] : max_x;

            int start_y = (offset_col > j) ? (offset_col - j) : 0;
            int max_y   = t->shape[1] - j + offset_col;
            int end_y   = (k->shape[1] < max_y) ? k->shape[1] : max_y;

            // Scorre le righe valide del kernel.
            for (int x = start_x; x < end_x; x++) {
                
                int curr_idx_row = i + x - offset_row;
                
                // Pre-calcola gli indici base per la riga corrente per evitare moltiplicazioni ripetute.
                int t_base_idx = curr_idx_row * t->shape[1] + j - offset_col;
                int k_base_idx = x * k->shape[1];

                // Scorre le colonne valide del kernel accumulando il prodotto dei valori.
                for (int y = start_y; y < end_y; y++) {
                    sum += t->values[t_base_idx + y] * k->values[k_base_idx + y];
                }
            }
            
            // Assegna il valore convoluto finale alla posizione corrispondente del tensore risultato.
            result->values[i * result->shape[1] + j] = sum;
        }   
    }

    return ERR_SUCCESS;
}

int tensor_reshape(tensor* t, tensor* s) {

    if ((t  ==  NULL)  ||  (s  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t);
    ASSERT_NUMERIC(s);

    int vector_result = is_vector(s);
    if (vector_result  !=  ERR_SUCCESS) return vector_result;

    int s_len = s->shape[0];
    // Verifica che la lunghezza della forma specificata sia supportata (1D o 2D).
    if ((s_len < 1)  ||  (s_len > 2)) {
        return ERR_SHAPE_MISMATCH; 
    }

    int new_shape0;
    int new_shape1;
    int new_ndim;

    // Determina le nuove dimensioni.
    if (s_len == 2) {
        new_shape0 = (int)s->values[0];
        new_shape1 = (int)s->values[1];
        new_ndim = 2;
    } else {
        new_shape0 = (int)s->values[0];
        new_shape1 = 0;
        new_ndim = 1;
    }

    // Calcola il numero totale di elementi del tensore di input.
    int t_shape;
    if (t->ndim == 1) {
        t_shape = t->shape[0];
    } else {
        t_shape = t->shape[0] * t->shape[1];
    }
    
    // Calcola il numero totale di elementi richiesti dalla nuova forma.
    int s_shape;
    if (new_ndim == 1) {
        s_shape = new_shape0;
    } else {
        s_shape = new_shape0 * new_shape1;
    }
    
    // Verifica che il numero totale di elementi rimanga invariato.
    if (t_shape  !=  s_shape) {
        return ERR_SHAPE_MISMATCH;
    }

    // Applica le nuove dimensioni e la nuova dimensionalità al tensore.
    t->ndim = new_ndim;
    t->shape[0] = new_shape0;
    t->shape[1] = new_shape1;

    return ERR_SUCCESS;
}

int tensor_ravel(tensor* t) {

    if (t  ==  NULL) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t);

    // Calcola il numero totale di elementi della struttura corrente.
    int t_shape;
    if (t->ndim == 1) {
        t_shape = t->shape[0];
    } else {
        t_shape = t->shape[0] * t->shape[1];
    }

    // Converte il tensore in un vettore 1D continuo.
    t->ndim = 1;
    t->shape[0] = t_shape;
    t->shape[1] = 0;

    return ERR_SUCCESS;
}

int tensor_get_shape(tensor* t, tensor* result) {

    if ((t  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t);

    int tensor_init_numeric_result;
    
    // Inizializza il tensore risultato come un vettore la cui dimensione dipende dal numero di dimensioni di 't'.
    if (t->ndim == 1) {
        int shape[] = {1};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {2};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    }
    
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    // Salva le dimensioni del tensore di input all'interno dei valori del vettore risultato.
    if (t->ndim == 1) {
        result->values[0] = (float)t->shape[0];
    } else {
        result->values[0] = (float)t->shape[0];
        result->values[1] = (float)t->shape[1];
    }

    return ERR_SUCCESS;
}

int tensor_fill(tensor* s, tensor* v, tensor* result) {

    if ((s  ==  NULL)  ||  (v  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(s);
    ASSERT_NUMERIC(v);

    int vector_result = is_vector(s);
    if (vector_result  !=  ERR_SUCCESS) return vector_result;

    int s_len = s->shape[0];

    // Verifica che la lunghezza della forma specificata sia supportata (1D o 2D).
    if ((s_len < 1)  ||  (s_len > 2)) {
        return ERR_SHAPE_MISMATCH; 
    }

    int new_shape0;
    int new_shape1;
    int new_ndim;

    // Estrae le nuove dimensioni (1D o 2D) dal vettore di forma.
    if (s_len == 2) {
        new_shape0 = (int)s->values[0];
        new_shape1 = (int)s->values[1];
        new_ndim = 2;
    } else {
        new_shape0 = (int)s->values[0];
        new_shape1 = 0;
        new_ndim = 1;
    }

    // Valida che le dimensioni estratte siano strettamente maggiori di zero.
    if (new_shape0 <= 0) {
        return ERR_SHAPE_MISMATCH;
    }
    
    if (new_ndim == 2) {
        if (new_shape1 <= 0) {
            return ERR_SHAPE_MISMATCH;
        }
    }

    // Determina il tipo di struttura del tensore in base alla dimensionalità.
    int tensor_shape;
    if (new_ndim == 1) {
        tensor_shape = TENSOR_SHAPE_VECTOR;
    } else {
        tensor_shape = TENSOR_SHAPE_MATRIX;
    }

    // Inizializza il tensore risultato con la forma calcolata.
    int shape[] = {new_shape0, new_shape1};
    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, tensor_shape);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    // Calcola il numero totale di elementi del tensore destinazione.
    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }
    
    // Calcola la lunghezza del tensore di valori sorgente.
    int v_len;
    if (v->ndim == 1) {
        v_len = v->shape[0];
    } else {
        v_len = v->shape[0] * v->shape[1];
    }

    // Verifica che la dimensione del target sia un multiplo della sorgente di riempimento.
    if (s_values % v_len != 0)  {
        return ERR_SHAPE_MISMATCH; 
    }

    // Riempie il tensore risultato ripetendo i valori del tensore sorgente in parallelo.
    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        int idx = i % v_len;
        result->values[i] = v->values[idx];
    }

    return ERR_SUCCESS;
}

int tensor_generate_random(tensor* s, tensor* result) {
    if ((s  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int vector_result = is_vector(s);
    if (vector_result  !=  ERR_SUCCESS) return vector_result;

    int s_len = s->shape[0];

    // Verifica che la lunghezza della forma specificata sia supportata (1D o 2D).
    if ((s_len < 1)  ||  (s_len > 2)) {
        return ERR_SHAPE_MISMATCH; 
    }

    int new_shape0;
    int new_shape1;
    int new_ndim;

    // Estrae le nuove dimensioni (1D o 2D) dal vettore di forma.
    if (s_len == 2) {
        new_shape0 = (int)s->values[0];
        new_shape1 = (int)s->values[1];
        new_ndim = 2;
    } else {
        new_shape0 = (int)s->values[0];
        new_shape1 = 0;
        new_ndim = 1;
    }

    // Valida che le dimensioni estratte siano strettamente maggiori di zero.
    if (new_shape0 <= 0) {
        return ERR_SHAPE_MISMATCH;
    }
    
    if (new_ndim == 2) {
        if (new_shape1 <= 0) {
            return ERR_SHAPE_MISMATCH;
        }
    }

    // Determina il tipo di struttura del tensore in base alla dimensionalità.
    int tensor_shape;
    if (new_ndim == 1) {
        tensor_shape = TENSOR_SHAPE_VECTOR;
    } else {
        tensor_shape = TENSOR_SHAPE_MATRIX;
    }

    // Inizializza il tensore risultato con la forma calcolata.
    int shape[] = {new_shape0, new_shape1};
    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, tensor_shape);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    // Calcola il numero totale di elementi del tensore risultato.
    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    // Pre-calcola il fattore di scala per normalizzare i valori casuali tra 0 e 1.
    float scale = 1.0f / (float)RAND_MAX;
    
    // Ottiene un seme iniziale dal generatore globale prima di entrare nella sezione parallela.
    unsigned int base_seed = rand(); 

    // Popola il tensore con numeri casuali in parallelo, assegnando un seme unico per ciascun thread.
    #pragma omp parallel 
    {
        unsigned int seed = base_seed + omp_get_thread_num(); 

        #pragma omp for
        for (int i = 0; i < s_values; i++){
            result->values[i] = (float)rand_r(&seed) * scale;
        }
    }

    return ERR_SUCCESS;
}