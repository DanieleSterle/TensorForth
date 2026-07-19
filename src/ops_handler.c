// DANIELE STERLE SM3201594

#include "ops_handler.h"
#include "ops_math.h"
#include "utils.h"
#include <stdlib.h>

int handle_add_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free(t1.values);
        return ERR_STACK_UNDERFLOW;
    }

    int add_result = tensor_add(&t1, &t2, &result);
    if (add_result != ERR_SUCCESS) {
        free(t1.values);
        free(t2.values);
        return add_result;
    }

    free(t1.values);
    free(t2.values);

    *s_head = push(stack, result, s_size, *s_head);
    return ERR_SUCCESS; // 0 means the whole stack operation succeeded
}

int handle_subtract_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free(t1.values);
        return ERR_STACK_UNDERFLOW;
    }

    int subtract_result = tensor_subtract(&t1, &t2, &result);
    if (subtract_result != ERR_SUCCESS) {
        free(t1.values);
        free(t2.values);
        return subtract_result;
    }

    free(t1.values);
    free(t2.values);

    *s_head = push(stack, result, s_size, *s_head);
    return ERR_SUCCESS; // 0 means the whole stack operation succeeded
}

int handle_multiply_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free(t1.values);
        return ERR_STACK_UNDERFLOW;
    }

    int multiply_result = tensor_multiply(&t1, &t2, &result);
    if (multiply_result != ERR_SUCCESS) {
        free(t1.values);
        free(t2.values);
        return multiply_result;
    }

    free(t1.values);
    free(t2.values);

    *s_head = push(stack, result, s_size, *s_head);
    return ERR_SUCCESS; // 0 means the whole stack operation succeeded
}

int handle_lt_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free(t1.values);
        return ERR_STACK_UNDERFLOW;
    }

    int multiply_result = tensor_lt(&t1, &t2, &result);
    if (multiply_result != ERR_SUCCESS) {
        free(t1.values);
        free(t2.values);
        return multiply_result;
    }

    free(t1.values);
    free(t2.values);

    *s_head = push(stack, result, s_size, *s_head);
    return ERR_SUCCESS; // 0 means the whole stack operation succeeded
}

int handle_gt_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free(t1.values);
        return ERR_STACK_UNDERFLOW;
    }

    int multiply_result = tensor_gt(&t1, &t2, &result);
    if (multiply_result != ERR_SUCCESS) {
        free(t1.values);
        free(t2.values);
        return multiply_result;
    }

    free(t1.values);
    free(t2.values);

    *s_head = push(stack, result, s_size, *s_head);
    return ERR_SUCCESS; // 0 means the whole stack operation succeeded
}

int handle_eq_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free(t1.values);
        return ERR_STACK_UNDERFLOW;
    }

    int multiply_result = tensor_eq(&t1, &t2, &result);
    if (multiply_result != ERR_SUCCESS) {
        free(t1.values);
        free(t2.values);
        return multiply_result;
    }

    free(t1.values);
    free(t2.values);

    *s_head = push(stack, result, s_size, *s_head);
    return ERR_SUCCESS; // 0 means the whole stack operation succeeded
}