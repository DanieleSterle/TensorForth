// Daniele Sterle SM3201594

// Element-wise mathematics and logic.
// Contains OpenMP-parallelized loops for arithmetic, comparisons, and logic.

#include "../include/ops_math.h"

// STRATEGY:
// check stack size
// Validate Dimensions
// Execute & push

// Signatures: int op(tensor* stack)
// print in main

/*

// TORNARE INTERO + PASSARE stack
tensor tensor_add(tensor t1, tensor t2) {

    if ((t1.rows  ==  t2.rows)  &&  (t1.columns  ==  t2.columns)){

        tensor res;
        res.columns = t1.columns;
        res.rows = t1.rows;

        for (int i = 0; i < res.rows; i++){
            for (int j = 0; j < res.columns; j++){
                res.values[i * res.columns + j] = t1.values[i * res.columns + j] + t1.values[i * res.columns + j];
            }
        }

        return res;
    }

    return;

}

*/