// Daniele Sterle SM3201594

#ifndef TENSORFORTH_OPS_MATH_H
#define TENSORFORTH_OPS_MATH_H

#include <math.h>
#include "utils.h"
#include "tensor.h"

int tensor_add(tensor* t1, tensor* t2, tensor* result);
int tensor_subtract(tensor* t1, tensor* t2, tensor* result);
int tensor_multiply(tensor* t1, tensor* t2, tensor* result);

int tensor_lt(tensor* t1, tensor* t2, tensor* result);
int tensor_gt(tensor* t1, tensor* t2, tensor* result);
int tensor_eq(tensor* t1, tensor* t2, tensor* result);

int tensor_and(tensor* t1, tensor* t2, tensor* result);
int tensor_or(tensor* t1, tensor* t2, tensor* result);
int tensor_not(tensor* t, tensor* result);

int tensor_select(tensor* t1, tensor* t2, tensor* mask, tensor* result);

int tensor_relu(tensor* t1, tensor* result);
int tensor_element_min(tensor* t1, tensor* t2, tensor* result);
int tensor_element_max(tensor* t1, tensor* t2, tensor* result);

int tensor_sum_reduce(tensor* t, tensor* result);

#endif