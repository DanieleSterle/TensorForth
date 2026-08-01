// Daniele Sterle SM3201594

// Disk operations and file formatting.
// Manages PGM image reading/writing and implements the 64-byte aligned on_disk_tensor struct.

#include "io.h"

int tensor_read_pgm(tensor* t, tensor* result) {
    if ((t  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_STRING(t);

    FILE* out_file = fopen(t->filename, "rb");
    if (out_file  ==  NULL) {
        // DEFINIRE ERRORE
        return -1;
    }

    // 3. Parse the PGM Header
    char magic[3];
    int width, height, maxval;
    
    // Read the "P5" identifier
    if ((fscanf(out_file, "%2s", magic)  !=  1)  ||  (magic[0]  !=  'P')  ||  (magic[1]  !=  '5')) {
        fclose(out_file);
        return ERR_IO; // Not a valid binary PGM
    }

    // (Note: A truly robust PGM parser skips comment lines starting with '#' here)
    
    // Read dimensions and max value
    if (fscanf(out_file, "%d %d %d", &width, &height, &maxval)  !=  3) {
        fclose(out_file);
        return ERR_IO;
    }

    // Consume the single whitespace character that separates the header from the binary data
    fgetc(out_file);

    // 4. Allocate memory for the raw 8-bit image data
    int num_pixels = width * height;
    unsigned char* raw_pixels = (unsigned char*) malloc(num_pixels * sizeof(unsigned char));
    if (raw_pixels  ==  NULL) {
        fclose(out_file);
        return ERR_OUT_OF_MEMORY;
    }

    // 5. Sequentially read the entire binary block at once (Fastest I/O method)
    if ((int) fread(raw_pixels, sizeof(unsigned char), num_pixels, out_file)  !=  num_pixels) {
        free(raw_pixels);
        fclose(out_file);
        return ERR_IO;
    }
    
    fclose(out_file);


    int create_res = create_numeric_tensor(result, NULL, height, width);
    if (create_res != ERR_SUCCESS) {
        free(raw_pixels);
        return create_res;
    }

    // OPTIMIZE
    for (int i = 0; i < num_pixels; i++) {
        result->values[i] = (float)raw_pixels[i] / 255.0f;
    }

    return ERR_SUCCESS;
}

int tensor_write_pgm(tensor* string_t, tensor* numeric_t) {
    if ((string_t  ==  NULL)  ||  (numeric_t  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_STRING(string_t);
    ASSERT_NUMERIC(numeric_t);

    int matrix_result = is_matrix(numeric_t);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    FILE* out_file = fopen(string_t->filename, "wb");
    if (out_file  ==  NULL) {
        // DEFINIRE ERRORE
        return -1;
    }

    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n", numeric_t->columns, numeric_t->rows);
    fprintf(out_file, "255\n");

    int num_pixels = numeric_t->rows * numeric_t->columns;
    unsigned char* pixels = (unsigned char*) malloc(num_pixels * sizeof(unsigned char));
    
    if (pixels == NULL) {
        fclose(out_file);
        return ERR_OUT_OF_MEMORY;
    }

    // OPTIMIZE
    for (int i = 0; i < num_pixels; i++) {
        float val = numeric_t->values[i];
        
        if (val < 0.0f) {
            val = 0.0f;
        } 

        else if (val > 1.0f) {
            val = 1.0f;
        }
        
        // "rimappati in [0, 255]"
        pixels[i] = (unsigned char)(val * 255.0f);
    }

    fwrite(pixels, sizeof(unsigned char), num_pixels, out_file);
    free(pixels);
    fclose(out_file);

    return ERR_SUCCESS;
}

int tensor_read_mmap(tensor* t, tensor* result) {
    if ((t  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_STRING(t);

    FILE* out_file = fopen(t->filename, "rb");
    if (out_file  ==  NULL) {
        // DEFINIRE ERRORE
        return -1;
    }

    int fd = fileno(out_file);
    if (fd < 0) {
        fclose(out_file);
        // DEFINIRE ERRORE
        return -1;
    }

    struct stat file_stat;
    if (fstat(fd, &file_stat) < 0) {
        close(fd);
        free_tensor(t);
        return ERR_IO;
    }

    float* mapped_data = mmap(NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_data  ==  MAP_FAILED) {
        fclose(out_file);
        free_tensor(t);
        return ERR_IO;
    }
    
    // OPTIMIZE w madvise ?

    fclose(out_file);

    on_disk_tensor* header = (on_disk_tensor*) mapped_data;

    result->type = TYPE_NUMERIC_MMAP;
    
    if (header->ndim == 1) {
        result->rows = 1;
        result->columns = header->shape[0];
    } else {
        result->rows = header->shape[0];
        result->columns = header->shape[1];
    }

    result->values = (float*) ((char*)mapped_data + header->data_offset);
    
    // Allocate the reference counter (this still uses malloc)
    result->ref_count = (int*) malloc(sizeof(int));
    if (result->ref_count == NULL) {
        munmap(mapped_data, file_stat.st_size); // Clean up on failure
        free_tensor(t);
        return ERR_OUT_OF_MEMORY;
    }
    *(result->ref_count) = 1;

    return ERR_SUCCESS;
}

int tensor_write_bin(tensor* string_t, tensor* numeric_t) {
    if ((string_t  ==  NULL)  ||  (numeric_t  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_STRING(string_t);
    ASSERT_NUMERIC(numeric_t);

    // int matrix_result = is_matrix(numeric_t);
    // if (matrix_result != ERR_SUCCESS) return matrix_result;
    
    FILE* out_file = fopen(string_t->filename, "wb");
    if (out_file == NULL) {
        return ERR_IO; 
    }

    on_disk_tensor header;
    if (numeric_t->rows == 1) {
        header.ndim = 1;
        header.shape[0] = numeric_t->columns;
        header.shape[1] = 0;
    } else {
        header.ndim = 2;
        header.shape[0] = numeric_t->rows;
        header.shape[1] = numeric_t->columns;
    }
    header.data_offset = sizeof(on_disk_tensor);

    size_t written_header = fwrite(&header, sizeof(on_disk_tensor), 1, out_file);
    if (written_header != 1) {
        fclose(out_file);
        return ERR_IO;
    }

    int num_elements = numeric_t->rows * numeric_t->columns;
    size_t written_data = fwrite(numeric_t->values, sizeof(float), num_elements, out_file);
    
    if ((int) written_data != num_elements) {
        fclose(out_file);
        return ERR_IO;
    }

    fclose(out_file);

    return ERR_SUCCESS;
}