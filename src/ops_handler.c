// DANIELE STERLE SM3201594

#include "ops_handler.h"
#include "ops_math.h"
#include "ops_tensor.h"
#include "tensor.h"
#include "utils.h"
#include <stdlib.h>

int handle_add_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int add_result = tensor_add(&t1, &t2, &result);
    if (add_result != ERR_SUCCESS) {
        free_tensor(&t1);
        free_tensor(&t2);
        return add_result;
    }

    free_tensor(&t1);
    free_tensor(&t2);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result; 
    return ERR_SUCCESS;
}

int handle_subtract_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int subtract_result = tensor_subtract(&t1, &t2, &result);
    if (subtract_result != ERR_SUCCESS) {
        free_tensor(&t1);
        free_tensor(&t2);
        return subtract_result;
    }

    free_tensor(&t1);
    free_tensor(&t2);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_multiply_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int multiply_result = tensor_multiply(&t1, &t2, &result);
    if (multiply_result != ERR_SUCCESS) {
        free_tensor(&t1);
        free_tensor(&t2);
        return multiply_result;
    }

    free_tensor(&t1);
    free_tensor(&t2);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_lt_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int lt_result = tensor_lt(&t1, &t2, &result);
    if (lt_result != ERR_SUCCESS) {
        free_tensor(&t1);
        free_tensor(&t2);
        return lt_result;
    }

    free_tensor(&t1);
    free_tensor(&t2);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_gt_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int gt_result = tensor_gt(&t1, &t2, &result);
    if (gt_result != ERR_SUCCESS) {
        free_tensor(&t1);
        free_tensor(&t2);
        return gt_result;
    }

    free_tensor(&t1);
    free_tensor(&t2);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_eq_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int eq_result = tensor_eq(&t1, &t2, &result);
    if (eq_result != ERR_SUCCESS) {
        free_tensor(&t1);
        free_tensor(&t2);
        return eq_result;
    }

    free_tensor(&t1);
    free_tensor(&t2);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_and_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int and_result = tensor_and(&t1, &t2, &result);
    if (and_result != ERR_SUCCESS) {
        free_tensor(&t1);
        free_tensor(&t2);
        return and_result;
    }

    free_tensor(&t1);
    free_tensor(&t2);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_or_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int or_result = tensor_or(&t1, &t2, &result);
    if (or_result != ERR_SUCCESS) {
        free_tensor(&t1);
        free_tensor(&t2);
        return or_result;
    }

    free_tensor(&t1);
    free_tensor(&t2);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_not_op(tensor** stack, int* s_size, int* s_head) {
    tensor t, result;

    *s_head = pop(*stack, &t, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    int not_result = tensor_not(&t, &result);
    if (not_result != ERR_SUCCESS) {
        free_tensor(&t);
        return not_result;
    }

    free_tensor(&t);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_select_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, mask, result;

    *s_head = pop(*stack, &mask, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&mask);
        return ERR_STACK_UNDERFLOW;
    }

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&mask);
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int select_result = tensor_select(&t1, &t2, &mask, &result);
    if (select_result != ERR_SUCCESS) {
        free_tensor(&mask);
        free_tensor(&t1);
        free_tensor(&t2);
        return select_result;
    }

    free_tensor(&mask);
    free_tensor(&t1);
    free_tensor(&t2);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_matmul_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int matmul_result = tensor_matmul(&t1, &t2, &result);
    if (matmul_result != ERR_SUCCESS) {
        free_tensor(&t1);
        free_tensor(&t2);
        return matmul_result;
    }

    free_tensor(&t1);
    free_tensor(&t2);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_dot_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int dot_result = tensor_dot(&t1, &t2, &result);
    if (dot_result != ERR_SUCCESS) {
        free_tensor(&t1);
        free_tensor(&t2);
        return dot_result;
    }

    free_tensor(&t1);
    free_tensor(&t2);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_conv2d_op(tensor** stack, int* s_size, int* s_head) {
    tensor t, k, result;

    *s_head = pop(*stack, &k, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&k);
        return ERR_STACK_UNDERFLOW;
    }

    int matmul_result = tensor_conv2d(&t, &k, &result);
    if (matmul_result != ERR_SUCCESS) {
        free_tensor(&t);
        free_tensor(&k);
        return matmul_result;
    }

    free_tensor(&t);
    free_tensor(&k);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_reshape_op(tensor** stack, int* s_size, int* s_head) {
    tensor t, s;

    *s_head = pop(*stack, &s, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&s);
        return ERR_STACK_UNDERFLOW;
    }

    int reshape_result = tensor_reshape(&t, &s);
    if (reshape_result != ERR_SUCCESS) {
        free_tensor(&t);
        free_tensor(&s);
        return reshape_result;
    }

    free_tensor(&s);

    int push_result = push(stack, t, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&t); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_ravel_op(tensor** stack, int* s_size, int* s_head) {
    tensor t;

    *s_head = pop(*stack, &t, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    int revel_result = tensor_ravel(&t);
    if (revel_result != ERR_SUCCESS) {
        free_tensor(&t);
        return revel_result;
    }

    int push_result = push(stack, t, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&t); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_get_shape_op(tensor** stack, int* s_size, int* s_head) {
    tensor t, result;

    *s_head = pop(*stack, &t, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    int get_shape_result = tensor_get_shape(&t, &result);
    if (get_shape_result != ERR_SUCCESS) {
        free_tensor(&t);
        return get_shape_result;
    }

    free_tensor(&t);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_random_op(tensor** stack, int* s_size, int* s_head) {
    tensor s, result;

    *s_head = pop(*stack, &s, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    int random_result = tensor_generate_random(&s, &result);
    if (random_result != ERR_SUCCESS) {
        free_tensor(&s);
        return random_result;
    }

    free_tensor(&s);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_relu_op(tensor** stack, int* s_size, int* s_head) {
    tensor t, result;

    *s_head = pop(*stack, &t, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    int relu_result = tensor_relu(&t, &result);
    if (relu_result != ERR_SUCCESS) {
        free_tensor(&t);
        return relu_result;
    }

    free_tensor(&t);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_min_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int min_result = tensor_element_min(&t1, &t2, &result);
    if (min_result != ERR_SUCCESS) {
        free_tensor(&t1);
        free_tensor(&t2);
        return min_result;
    }

    free_tensor(&t1);
    free_tensor(&t2);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_max_op(tensor** stack, int* s_size, int* s_head) {
    tensor t1, t2, result;

    *s_head = pop(*stack, &t1, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &t2, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&t1);
        return ERR_STACK_UNDERFLOW;
    }

    int max_result = tensor_element_max(&t1, &t2, &result);
    if (max_result != ERR_SUCCESS) {
        free_tensor(&t1);
        free_tensor(&t2);
        return max_result;
    }

    free_tensor(&t1);
    free_tensor(&t2);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_sum_reduce_op(tensor** stack, int* s_size, int* s_head) {
    tensor t, result;

    *s_head = pop(*stack, &t, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    int reduce_result = tensor_sum_reduce(&t, &result);
    if (reduce_result != ERR_SUCCESS) {
        free_tensor(&t);
        return reduce_result;
    }

    free_tensor(&t);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_fill_op(tensor** stack, int* s_size, int* s_head) {
    tensor s, v, result;

    *s_head = pop(*stack, &v, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    *s_head = pop(*stack, &s, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) {
        free_tensor(&v);
        return ERR_STACK_UNDERFLOW;
    }

    int add_result = tensor_fill(&s, &v, &result);
    if (add_result != ERR_SUCCESS) {
        free_tensor(&v);
        free_tensor(&s);
        return add_result;
    }

    free_tensor(&v);
    free_tensor(&s);

    int push_result = push(stack, result, s_size, *s_head);
    
    if (push_result < 0) { 
        free_tensor(&result); 
        return push_result;  
    }

    *s_head = push_result;
    return ERR_SUCCESS;
}

int handle_print_op(tensor** stack, int* s_head) {
    tensor t;

    *s_head = pop(*stack, &t, *s_head);
    if (*s_head == ERR_STACK_UNDERFLOW) return ERR_STACK_UNDERFLOW;

    tensor_print(&t);
    free_tensor(&t);

    return ERR_SUCCESS;
}