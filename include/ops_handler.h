// DANIELE STERLE SM3201594

#include "stack.h"
#include "io.h"

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

int handle_reshape_op(tensor** stack, int* s_size, int* s_head);
int handle_ravel_op(tensor** stack, int* s_size, int* s_head);
int handle_get_shape_op(tensor** stack, int* s_size, int* s_head);

int handle_random_op(tensor** stack, int* s_size, int* s_head);

int handle_relu_op(tensor** stack, int* s_size, int* s_head);
int handle_min_op(tensor** stack, int* s_size, int* s_head);
int handle_max_op(tensor** stack, int* s_size, int* s_head);

int handle_sum_reduce_op(tensor** stack, int* s_size, int* s_head);

int handle_fill_op(tensor** stack, int* s_size, int* s_head);

int handle_print_op(tensor** stack, int* s_head);

int handle_read_pgm_op(tensor** stack, int* s_size, int* s_head);
int handle_write_pgm_op(tensor** stack, int* s_head);
int handle_read_mmap_op(tensor** stack, int* s_size, int* s_head);
int handle_write_bin_op(tensor** stack, int* s_head);

