// Daniele Sterle SM3201594

#ifndef TENSORFORTH_UTILS_H
#define TENSORFORTH_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include "stack.h"

void free_all(FILE* fd, tensor* stack, int s_stack, float* values);

#endif