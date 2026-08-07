// Daniele Sterle SM3201594

// Disk operations and file formatting.
// Manages PGM image reading/writing and implements the 64-byte aligned on_disk_tensor struct.

#include "io.h"

// Skips whitespaces and any comment lines starting with '#'
void skip_pgm_comments(FILE *fp) {
    int ch;
    while (1) {
        // Skip leading whitespaces
        do {
            ch = fgetc(fp);
        } while (isspace(ch));

        if (ch == EOF) {
            break;
        }

        // If a comment is found, skip characters until the end of the line
        if (ch == '#') {
            do {
                ch = fgetc(fp);
            } while (ch != '\n' && ch != EOF);
        } else {
            // Not a comment and not a space, put the character back for fscanf
            ungetc(ch, fp);
            break;
        }
    }
}

int tensor_read_pgm(tensor* t, tensor* result) {
    if ((t  ==  NULL)  ||  (result  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_STRING(t);

    FILE* out_file = fopen(t->filename, "rb");
    if (out_file  ==  NULL) {
        return ERR_FILE_OPEN;
    }

    char magic[3];
    int width, height, maxval;
    
    // Read the "P5" identifier
    if ((fscanf(out_file, "%2s", magic)  !=  1)  ||  (magic[0]  !=  'P')  ||  (magic[1]  !=  '5')) {
        fclose(out_file);
        return ERR_INVALID_PGM;
    }

    skip_pgm_comments(out_file);
    if (fscanf(out_file, "%d", &width) != 1) {
        fclose(out_file);
        return ERR_IO;
    }

    // Skip comments/whitespace before height
    skip_pgm_comments(out_file);
    if (fscanf(out_file, "%d", &height) != 1) {
        fclose(out_file);
        return ERR_IO;
    }

    // Skip comments/whitespace before maxval
    skip_pgm_comments(out_file);
    if (fscanf(out_file, "%d", &maxval) != 1) {
        fclose(out_file);
        return ERR_IO;
    }

    // Consume the single whitespace character that separates the header from the binary data
    fgetc(out_file);

    int num_pixels = width * height;
    unsigned char* raw_pixels = (unsigned char*) malloc(num_pixels * sizeof(unsigned char));
    if (raw_pixels  ==  NULL) {
        fclose(out_file);
        return ERR_OUT_OF_MEMORY;
    }

    if ((int) fread(raw_pixels, sizeof(unsigned char), num_pixels, out_file)  !=  num_pixels) {
        free(raw_pixels);
        fclose(out_file);
        return ERR_IO;
    }
    
    fclose(out_file);

    int shape[] = {height, width};

    int create_res = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    if (create_res != ERR_SUCCESS) {
        free(raw_pixels);
        return create_res;
    }

    #pragma omp parallel for
    for (int i = 0; i < num_pixels; i++) {
        result->values[i] = (float)raw_pixels[i] / 255.0f;
    }

    free(raw_pixels);
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
        return ERR_FILE_OPEN;
    }

    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n", numeric_t->shape[1], numeric_t->shape[0]);
    fprintf(out_file, "255\n");

    int num_pixels = numeric_t->shape[0] * numeric_t->shape[1];
    unsigned char* pixels = (unsigned char*) malloc(num_pixels * sizeof(unsigned char));
    
    if (pixels == NULL) {
        fclose(out_file);
        return ERR_OUT_OF_MEMORY;
    }

    #pragma omp parallel for
    for (int i = 0; i < num_pixels; i++) {
        float val = numeric_t->values[i];        
        val = fminf(fmaxf(val, 0.0f), 1.0f);
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
        return ERR_FILE_OPEN;
    }

    int fd = fileno(out_file);
    if (fd < 0) {
        fclose(out_file);
        return ERR_IO;
    }

    struct stat file_stat;
    if (fstat(fd, &file_stat) < 0) {
        close(fd);
        free_tensor(t);
        return ERR_FILE_STAT;
    }

    float* mapped_data = mmap(NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_data  ==  MAP_FAILED) {
        fclose(out_file);
        free_tensor(t);
        return ERR_MMAP_FAILED;
    }

    madvise(mapped_data, file_stat.st_size, MADV_WILLNEED);
    
    fclose(out_file);

    on_disk_tensor* header = (on_disk_tensor*) mapped_data;

    result->type = TYPE_NUMERIC_MMAP;
    result->ndim = header->ndim;
    
    if (header->ndim == 1) {
        result->shape[0] = header->shape[0];
        result->shape[1] = 0;
    } else {
        result->shape[0] = header->shape[0];
        result->shape[1] = header->shape[1];
    }

    result->values = (float*) ((char*)mapped_data + header->data_offset);
    
    result->ref_count = (int*) malloc(sizeof(int));
    if (result->ref_count == NULL) {
        munmap(mapped_data, file_stat.st_size);
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
    
    FILE* out_file = fopen(string_t->filename, "wb");
    if (out_file == NULL) {
        return ERR_FILE_OPEN; 
    }

    on_disk_tensor header = {0};

    if (numeric_t->ndim == 1) {
        header.ndim = 1;
        header.shape[0] = numeric_t->shape[0];
        header.shape[1] = 0;
    } else {
        header.ndim = 2;
        header.shape[0] = numeric_t->shape[0];
        header.shape[1] = numeric_t->shape[1];
    }

    header.data_offset = 64;

    size_t written_header = fwrite(&header, sizeof(on_disk_tensor), 1, out_file);
    if (written_header != 1) {
        fclose(out_file);
        return ERR_IO;
    }

    int padding_size = 64 - sizeof(on_disk_tensor);
    if (padding_size > 0) {
        char padding[64] = {0}; // Array of zeros
        fwrite(padding, sizeof(char), padding_size, out_file);
    }
    
    int num_elements = numeric_t->shape[0];
    if (numeric_t->ndim == 2) {
        num_elements *= numeric_t->shape[1];
    }

    size_t written_data = fwrite(numeric_t->values, sizeof(float), num_elements, out_file);
    
    if ((int) written_data != num_elements) {
        fclose(out_file);
        return ERR_IO;
    }

    fclose(out_file);

    return ERR_SUCCESS;
}