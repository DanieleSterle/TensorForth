// Daniele Sterle SM3201594

#ifndef TENSORFORTH_OPS_TENSOR_H
#define TENSORFORTH_OPS_TENSOR_H

#include "stack.h"

int tensor_matmul(tensor* t1, tensor* t2, tensor* result);
int tensor_dot(tensor* t1, tensor* t2, tensor* result);
int tensor_conv2d(tensor* t1, tensor* t2, tensor* result);

#endif