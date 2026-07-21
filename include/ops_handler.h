// DANIELE STERLE SM3201594

#include "stack.h"

int handle_add_op(tensor** stack, int* s_size, int* s_head);
int handle_subtract_op(tensor** stack, int* s_size, int* s_head);
int handle_multiply_op(tensor** stack, int* s_size, int* s_head);

int handle_lt_op(tensor** stack, int* s_size, int* s_head);
int handle_gt_op(tensor** stack, int* s_size, int* s_head);
int handle_eq_op(tensor** stack, int* s_size, int* s_head);

int handle_and_op(tensor** stack, int* s_size, int* s_head);
int handle_or_op(tensor** stack, int* s_size, int* s_head);
int handle_not_op(tensor** stack, int* s_size, int* s_head);

int handle_select_op(tensor** stack, int* s_size, int* s_head);

int handle_matmul_op(tensor** stack, int* s_size, int* s_head);
int handle_dot_op(tensor** stack, int* s_size, int* s_head);
int handle_conv2d_op(tensor** stack, int* s_size, int* s_head);