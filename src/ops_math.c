// Daniele Sterle SM3201594

// Element-wise mathematics and logic.
// Contains OpenMP-parallelized loops for arithmetic, comparisons, and logic.

#include "ops_math.h"
#include "utils.h"

int tensor_add(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int create_tensor_result = create_tensor(result, NULL, t1->rows, t1->columns);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = result->rows * result->columns;

    // OPTIMIZE
    for (int i = 0; i < s_values; i++) {
        result->values[i] = t1->values[i] + t2->values[i];
    }

    return ERR_SUCCESS;

}

int tensor_subtract(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int create_tensor_result = create_tensor(result, NULL, t1->rows, t1->columns);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = result->rows * result->columns;

    // OPTIMIZE
    for (int i = 0; i < s_values; i++) {
        result->values[i] = t1->values[i] - t2->values[i];
    }

    return ERR_SUCCESS;

}

int tensor_multiply(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int create_tensor_result = create_tensor(result, NULL, t1->rows, t1->columns);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = result->rows * result->columns;

    // OPTIMIZE
    for (int i = 0; i < s_values; i++) {
        result->values[i] = t1->values[i] * t2->values[i];
    }

    return ERR_SUCCESS;

}

int tensor_lt(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int create_tensor_result = create_tensor(result, NULL, t1->rows, t1->columns);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = result->rows * result->columns;

    // OPTIMIZE
    for (int i = 0; i < s_values; i++) {
        result->values[i] = (t1->values[i] < t2->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_gt(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int create_tensor_result = create_tensor(result, NULL, t1->rows, t1->columns);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = result->rows * result->columns;

    // OPTIMIZE
    for (int i = 0; i < s_values; i++) {
        result->values[i] = (t1->values[i] > t2->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_eq(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int create_tensor_result = create_tensor(result, NULL, t1->rows, t1->columns);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = result->rows * result->columns;

    // OPTIMIZE
    for (int i = 0; i < s_values; i++) {
        result->values[i] = (t1->values[i] == t2->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_and(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int is_bool;
    is_bool = is_boolean(t1);
    if (is_bool != ERR_SUCCESS) return is_bool;

    is_bool = is_boolean(t2);
    if (is_bool != ERR_SUCCESS) return is_bool;

    int create_tensor_result = create_tensor(result, NULL, t1->rows, t1->columns);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = result->rows * result->columns;

    // OPTIMIZE
    for (int i = 0; i < s_values; i++) {
        result->values[i] = (float)(t1->values[i] && t2->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_or(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int is_bool;
    is_bool = is_boolean(t1);
    if (is_bool != ERR_SUCCESS) return is_bool;

    is_bool = is_boolean(t2);
    if (is_bool != ERR_SUCCESS) return is_bool;

    int create_tensor_result = create_tensor(result, NULL, t1->rows, t1->columns);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = result->rows * result->columns;

    // OPTIMIZE
    for (int i = 0; i < s_values; i++) {
        result->values[i] = (float)(t1->values[i] || t2->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_not(tensor* t, tensor* result) {

    if ((t  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int is_bool = is_boolean(t);
    if (is_bool != ERR_SUCCESS) return is_bool;

    int create_tensor_result = create_tensor(result, NULL, t->rows, t->columns);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = result->rows * result->columns;

    // OPTIMIZE
    for (int i = 0; i < s_values; i++) {
        result->values[i] = (float)(!t->values[i]);
    }

    return ERR_SUCCESS;

}

int tensor_select(tensor* t1, tensor* t2, tensor* mask, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result;
    shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    shape_result = shape_cmp(t1, mask);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int is_bool = is_boolean(mask);
    if (is_bool != ERR_SUCCESS) return is_bool;

    int create_tensor_result = create_tensor(result, NULL, t1->rows, t1->columns);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = result->rows * result->columns;

    // OPTIMIZE
    for (int i = 0; i < s_values; i++) {
        result->values[i] = mask->values[i] * t1->values[i] + (1 - mask->values[i]) * t2->values[i];
    }

    return ERR_SUCCESS;

}

int tensor_relu(tensor* t, tensor* result) {

    if ((t  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int create_tensor_result = create_tensor(result, NULL, t->rows, t->columns);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = result->rows * result->columns;

    // OPTIMIZE
    for (int i = 0; i < s_values; i++) {
        // Barnchless?
        if (t->values[i] > 0.0f) {
            result->values[i] = t->values[i];
        } else {
            result->values[i] = 0.0f;
        }
    }

    return ERR_SUCCESS;

}

int tensor_element_min(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int create_tensor_result = create_tensor(result, NULL, t1->rows, t1->columns);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = result->rows * result->columns;

    for (int i = 0; i < s_values; i++) {
        // Branchless?
        if (t1->values[i] < t2->values[i]) {
            result->values[i] = t1->values[i];
        } else {
            result->values[i] = t2->values[i];
        }
    }

    return ERR_SUCCESS;

}

int tensor_element_max(tensor* t1, tensor* t2, tensor* result) {

    if ((t1  ==  NULL)  ||  (t2  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int shape_result = shape_cmp(t1, t2);
    if (shape_result != ERR_SUCCESS) return shape_result;

    int create_tensor_result = create_tensor(result, NULL, t1->rows, t1->columns);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = result->rows * result->columns;

    
    for (int i = 0; i < s_values; i++) {
        // Branchless?
        if (t1->values[i] > t2->values[i]) {
            result->values[i] = t1->values[i];
        } else {
            result->values[i] = t2->values[i];
        }
    }

    return ERR_SUCCESS;

}

int tensor_sum_reduce(tensor* t, tensor* result) {

    if ((t  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    int create_tensor_result = create_tensor(result, NULL, 1, 1);
    if (create_tensor_result  !=  ERR_SUCCESS) {
        return create_tensor_result;
    }

    int s_values = t->rows * t->columns;
    result->values[0] = 0;

    // OPTIMIZE 
    for (int i = 0; i < s_values; i++) {
        result->values[0] += t->values[i];
    }

    return ERR_SUCCESS;

}