# TensorForth Interpreter

An interpreter for a stack-based language operating on tensors. Built in C and optimized with OpenMP. 

This is the first of two projects for the *Programmazione Avanzata e Parallela* course (Artificial Intelligence & Data Analytics B.Sc.) at Università di Trieste for the 2025/2026 academic year.

## Description

### How it works
The interpreter processes the input file provided via the CLI, pushing tensors onto the stack and executing operations in postfix order. Before executing an operator, it verifies that sufficient and compatible operands are available on the stack. Successful operations push their results back onto the stack, while errors (such as missing operands or dimension mismatches) cause the program to terminate with an error code. 

To efficiently handle large amounts of data, tensor operations are optimized using OpenMP.

### Supported Operations
TensorForth supports various types of operations, such as:
* Arithmetic & Logic element-wise
* Tensor Math
* Shape Manipulation
* Utility & I/O
* Stack Operations

> **Note:** For a complete and exhaustive documentation of all operators, refer to `project_C.pdf` at the bottom of this README.

## Tools Used

* **C:** Core programming language.
* **OpenMP:** API for shared-memory multiprocessing programming in C.
* **Valgrind:** Instrumentation framework for building dynamic analysis tools (used for memory leak detection).
* **GCC / Makefile:** Compilation and build automation.

## Implementation Notes

### Tensor Structure
To ensure an optimal structure, the `tensor` struct follows the tagged union pattern. Specifically, it uses an `enum` to identify the tensor type (e.g., numeric, numeric_mmap, string) and a `union` to store the corresponding data (e.g., an array of floats or a string). This design ensures that the structs are as compact as possible, achieving better spatial locality and improving the overall performance of the program.

### Operations Structure 
All operations follow a strict execution pipeline: `interpreter` → `handler` → `function`

Where:
* **interpreter:** Reads the operator token and calls its respective handler.
* **handler:** Manages the surrounding logic: popping operands from the stack, handling stack-level error checking, managing reference counts/deallocations, and pushing the final result.
* **function:** Executes the core operation, validating tensor shapes, types, and dimensions before computing.

This architecture ensures that each function is atomic (i.e., handles a single specific responsibility), keeping the code readable and maintainable in the long term.

### Error Propagation
Thanks to the architecture explained in the previous point, error codes are propagated from their source directly up to the main function. The main function then calls a dedicated utility in `utils.c` to print the errors to the screen.

### Optimization of the `?` Operation (Random Tensor)
Even though the project specifications advised against parallelizing the random generation operator due to thread-safety concerns with standard `rand()`, this operation was successfully optimized using OpenMP. 

To achieve this safely, the following approach was implemented:
* **Shared Base Seed:** A single base seed is generated before parallelization and shared across all threads.
* **Thread-Specific Seeds:** Each thread adds its unique identifier (retrieved via `omp_get_thread_num()`) to the base seed, ensuring every thread has a unique and independent starting point.
* **Thread-Safe Generation:** The `rand_r()` function is utilized to generate the numbers, providing a fully thread-safe alternative to standard `rand()`.

### Memory Management
All memory is strictly and correctly deallocated. Zero memory leaks have been verified using Valgrind.

## Project Structure

```text
.
├── examples/                  # Example programs provided with the project
├── include/                   # Header files (.h)
│   ├── common.h               # Common definitions
│   ├── interpreter.h          # Token parsing and execution declarations
│   ├── io.h                   # File I/O declarations
│   ├── ops_handler.h          # Operation handler declarations
│   ├── ops_math.h             # Mathematical operations declarations
│   ├── ops_tensor.h           # Tensor-specific operations declarations
│   ├── stack.h                # Stack management declarations
│   ├── tensor.h               # Tensor struct and tagged union definitions
│   └── utils.h                # Utility function declarations
├── src/                       # Source code files (.c)
│   ├── interpreter.c          # Token parsing and operation dispatching logic
│   ├── io.c                   # File reading and writing implementation
│   ├── ops_handler.c          # Stack manipulation and operand validation logic
│   ├── ops_math.c             # Mathematical operations implementation
│   ├── ops_tensor.c           # Tensor shape and manipulation implementation
│   ├── stack.c                # Stack logic (push, pop, etc.)
│   ├── tensor.c               # Tensor initialization and memory management
│   └── utils.c                # Error formatting and screen output utilities
├── tensorforth_tests/         # Test files and scripts
├── .gitignore                 # Git ignore rules
├── main.c                     # Entry point, orchestrates execution and final error handling
├── Makefile                   # Build instructions and compilation flags
├── project_C.pdf              # Official project documentation
└── README.md                  # This file
```

## Documentation

To better understand all the project’s specifications and details, check the official documentation:
[📄 Project Documentation (PDF)](./project_C.pdf)

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.