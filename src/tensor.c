// Daniele Sterle SM3201594

// Tensor data structures and lifecycle.
// Handles memory allocation, reference counting, and shape operations like r, \, and #.

#include <stdint.h>
#include <sys/types.h>
#define MAX_DIM 2

struct on_disk_tensor {
    int32_t shape[MAX_DIM];
    int32_t ndim;
    off_t data_offset;
};