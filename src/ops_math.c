// Daniele Sterle SM3201594

// Le funzioni presenti in questo modulo seguono tutte la stessa struttura: 
// controllo dei puntatori e dei tipi, verifica della compatibilità delle 
// dimensioni, inizializzazione del tensore risultato ed esecuzione 
// dell'operazione element-wise tramite OpenMP. 

// Per evitare di ripetere gli stessi commenti in ogni funzione, viene 
// descritta in dettaglio solamente la prima (tensor_add). Le funzioni 
// successive seguono lo stesso schema, variando esclusivamente 
// l'operazione applicata agli elementi.

#include "ops_math.h"

int tensor_add(tensor* t1, tensor* t2, tensor* result) {

    // Verifica che i puntatori ai due tensori di input e al tensore risultato 
    // siano validi. In caso contrario, restituisce un errore di puntatore nullo
    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    // Verifica che entrambi i tensori contengano dati di tipo numerico, 
    // requisito necessario per poter eseguire l'operazione di somma
    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    // Controlla che i due tensori abbiano la stessa forma
    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    // Inizializza il tensore risultato in base al numero di dimensioni (vettore o matrice)
    int tensor_init_numeric_result;
    if (t1->ndim == 1) {
        int shape[] = {t1->shape[0]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {t1->shape[0], t1->shape[1]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    }

    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    // Calcola il numero totale di elementi presenti nel tensore risultato.
    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    // Esegue la somma elemento per elemento in parallelo.
    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        result->values[i] = t1->values[i] + t2->values[i];
    }

    return ERR_SUCCESS;

}

int tensor_subtract(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int tensor_init_numeric_result;
    if (t1->ndim == 1) {
        int shape[] = {t1->shape[0]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {t1->shape[0], t1->shape[1]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    }

    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        result->values[i] = t1->values[i] - t2->values[i];
    }

    return ERR_SUCCESS;

}

int tensor_multiply(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int tensor_init_numeric_result;
    if (t1->ndim == 1) {
        int shape[] = {t1->shape[0]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {t1->shape[0], t1->shape[1]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    }

    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        result->values[i] = t1->values[i] * t2->values[i];
    }

    return ERR_SUCCESS;

}

int tensor_lt(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int tensor_init_numeric_result;
    if (t1->ndim == 1) {
        int shape[] = {t1->shape[0]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {t1->shape[0], t1->shape[1]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    }

    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        result->values[i] = (t1->values[i] < t2->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_gt(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int tensor_init_numeric_result;
    if (t1->ndim == 1) {
        int shape[] = {t1->shape[0]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {t1->shape[0], t1->shape[1]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    }

    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        result->values[i] = (t1->values[i] > t2->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_eq(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int tensor_init_numeric_result;
    if (t1->ndim == 1) {
        int shape[] = {t1->shape[0]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {t1->shape[0], t1->shape[1]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    }

    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        result->values[i] = (t1->values[i] == t2->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_and(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int is_bool;
    is_bool = is_boolean(t1);
    if (is_bool != ERR_SUCCESS) return is_bool;

    is_bool = is_boolean(t2);
    if (is_bool != ERR_SUCCESS) return is_bool;

    int tensor_init_numeric_result;
    if (t1->ndim == 1) {
        int shape[] = {t1->shape[0]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {t1->shape[0], t1->shape[1]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    }

    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        result->values[i] = (float)(t1->values[i] && t2->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_or(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int is_bool;
    is_bool = is_boolean(t1);
    if (is_bool != ERR_SUCCESS) return is_bool;

    is_bool = is_boolean(t2);
    if (is_bool != ERR_SUCCESS) return is_bool;

    int tensor_init_numeric_result;
    if (t1->ndim == 1) {
        int shape[] = {t1->shape[0]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {t1->shape[0], t1->shape[1]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    }

    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        result->values[i] = (float)(t1->values[i] || t2->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_not(tensor* t, tensor* result) {

    if ((t  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t);

    int is_bool = is_boolean(t);
    if (is_bool != ERR_SUCCESS) return is_bool;

    int tensor_init_numeric_result;
    if (t->ndim == 1) {
        int shape[] = {t->shape[0]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {t->shape[0], t->shape[1]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    }

    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        result->values[i] = (float)(!t->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_select(tensor* t1, tensor* t2, tensor* mask, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int shape_result;
    shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    shape_result = shape_cmp(t1, mask);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int is_bool = is_boolean(mask);
    if (is_bool != ERR_SUCCESS) return is_bool;

    int tensor_init_numeric_result;
    if (t1->ndim == 1) {
        int shape[] = {t1->shape[0]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {t1->shape[0], t1->shape[1]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    }

    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        result->values[i] = mask->values[i] * t1->values[i] + (1 - mask->values[i]) * t2->values[i];
    }

    return ERR_SUCCESS;

}

int tensor_relu(tensor* t, tensor* result) {

    if ((t  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t);

    int tensor_init_numeric_result;
    if (t->ndim == 1) {
        int shape[] = {t->shape[0]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {t->shape[0], t->shape[1]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    }

    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        result->values[i] = fmaxf(t->values[i], 0.0f);
    }

    return ERR_SUCCESS;

}

int tensor_element_min(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int tensor_init_numeric_result;
    if (t1->ndim == 1) {
        int shape[] = {t1->shape[0]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {t1->shape[0], t1->shape[1]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    }

    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        result->values[i] = fminf(t1->values[i], t2->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_element_max(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t1);
    ASSERT_NUMERIC(t2);

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int tensor_init_numeric_result;
    if (t1->ndim == 1) {
        int shape[] = {t1->shape[0]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    } else {
        int shape[] = {t1->shape[0], t1->shape[1]};
        tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    }

    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }
    
    int s_values;
    if (result->ndim == 1) {
        s_values = result->shape[0];
    } else {
        s_values = result->shape[0] * result->shape[1];
    }

    #pragma omp parallel for
    for (int i = 0; i < s_values; i++) {
        result->values[i] = fmaxf(t1->values[i], t2->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_sum_reduce(tensor* t, tensor* result) {

    if ((t  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_NUMERIC(t);

    int shape[] = {1};
    int tensor_init_numeric_result = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_VECTOR);
    if (tensor_init_numeric_result  !=  ERR_SUCCESS) {
        return tensor_init_numeric_result;
    }

    int s_values;
    if (t->ndim == 1) {
        s_values = t->shape[0];
    } else {
        s_values = t->shape[0] * t->shape[1];
    }
    
    float sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < s_values; i++) {
        sum += t->values[i];
    }

    result->values[0] = sum;

    return ERR_SUCCESS;

}