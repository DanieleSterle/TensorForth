// Daniele Sterle SM3201594

#ifndef TENSORFORTH_OPS_MATH_H
#define TENSORFORTH_OPS_MATH_H

#include "stack.h"

int tensor_add(tensor* t1, tensor* t2, tensor* result);
int tensor_subtract(tensor* t1, tensor* t2, tensor* result);
int tensor_multiply(tensor* t1, tensor* t2, tensor* result);

int tensor_lt(tensor* t1, tensor* t2, tensor* result);
int tensor_gt(tensor* t1, tensor* t2, tensor* result);
int tensor_eq(tensor* t1, tensor* t2, tensor* result);


#endif