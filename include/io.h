// Daniele Sterle SM3201594

#ifndef TENSORFORTH_IO_H
#define TENSORFORTH_IO_H

#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "utils.h"
#include "tensor.h"

int tensor_read_pgm(tensor* t, tensor* result);
int tensor_write_pgm(tensor* string_t, tensor* numeric_t);
int tensor_read_mmap(tensor* t, tensor* result);
int tensor_write_bin(tensor* string_t, tensor* numeric_t);

#endif