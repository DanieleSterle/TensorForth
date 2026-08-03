// Daniele Sterle SM3201594

#ifndef TENSORFORTH_OPS_TENSOR_H
#define TENSORFORTH_OPS_TENSOR_H

#include "utils.h"
#include "tensor.h"

int tensor_matmul(tensor* t1, tensor* t2, tensor* result);
int tensor_dot(tensor* t1, tensor* t2, tensor* result);
int tensor_conv2d(tensor* t1, tensor* t2, tensor* result);

int tensor_reshape(tensor* t, tensor* s);
int tensor_ravel(tensor* t);
int tensor_get_shape(tensor* t, tensor* result);

int tensor_fill(tensor* s, tensor* v, tensor* result);

int tensor_generate_random(tensor* s, tensor* result);

#endif