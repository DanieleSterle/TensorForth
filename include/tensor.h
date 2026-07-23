// Daniele Sterle SM3201594

#ifndef TENSORFORTH_TENSOR_H
#define TENSORFORTH_TENSOR_H


#include "utils.h"
#include "stack.h"

#include <stdint.h>
#include <sys/types.h>
#define MAX_DIM 2

struct on_disk_tensor {
    int32_t shape[MAX_DIM];
    int32_t ndim;
    off_t data_offset;
};

int tensor_reshape(tensor* t, tensor* s);
int tensor_ravel(tensor* t);
int tensor_get_shape(tensor* t, tensor* result);


#endif