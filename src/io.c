// Daniele Sterle SM3201594

// Disk operations and file formatting.
// Manages PGM image reading/writing and implements the 64-byte aligned on_disk_tensor struct.

#include "io.h"

// Salta gli spazi bianchi e le eventuali linee di commento che iniziano con '#'
void skip_pgm_comments(FILE *fp) {
    int ch;
    while (1) {
        // Salta gli spazi iniziali
        do {
            ch = fgetc(fp);
        } while (isspace(ch));

        if (ch == EOF) {
            break;
        }

        // Se viene trovato un commento, scarta i caratteri fino a fine linea
        if (ch == '#') {
            do {
                ch = fgetc(fp);
            } while (ch != '\n' && ch != EOF);
        } else {
            // Non è un commento né uno spazio: restituisce il carattere nello stream per fscanf
            ungetc(ch, fp);
            break;
        }
    }
}

// Legge un'immagine in formato PGM binario (P5, scala di grigi) come tensore 2D normalizzato in [0, 1]
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
    
    // Lettura e verifica dell'identificativo "P5" del formato PGM
    if ((fscanf(out_file, "%2s", magic)  !=  1)  ||  (magic[0]  !=  'P')  ||  (magic[1]  !=  '5')) {
        fclose(out_file);
        return ERR_INVALID_PGM;
    }

    // Acquisizione delle dimensioni (larghezza e altezza) e del valore massimo dei pixel
    skip_pgm_comments(out_file);
    if (fscanf(out_file, "%d", &width) != 1) {
        fclose(out_file);
        return ERR_IO;
    }

    skip_pgm_comments(out_file);
    if (fscanf(out_file, "%d", &height) != 1) {
        fclose(out_file);
        return ERR_IO;
    }

    skip_pgm_comments(out_file);
    if (fscanf(out_file, "%d", &maxval) != 1) {
        fclose(out_file);
        return ERR_IO;
    }

    // Consuma il singolo carattere di spaziatura che separa l'intestazione dai dati binari
    fgetc(out_file);

    int num_pixels = width * height;
    unsigned char* raw_pixels = (unsigned char*) malloc(num_pixels * sizeof(unsigned char));
    if (raw_pixels  ==  NULL) {
        fclose(out_file);
        return ERR_OUT_OF_MEMORY;
    }

    // Lettura dei byte grezzi dei pixel dal file
    if ((int) fread(raw_pixels, sizeof(unsigned char), num_pixels, out_file)  !=  num_pixels) {
        free(raw_pixels);
        fclose(out_file);
        return ERR_IO;
    }
    
    fclose(out_file);

    // Inizializzazione del tensore numerico di destinazione con forma 2D (matrice)
    int shape[] = {height, width};
    int create_res = tensor_init_numeric(result, NULL, shape, TENSOR_SHAPE_MATRIX);
    if (create_res != ERR_SUCCESS) {
        free(raw_pixels);
        return create_res;
    }

    // Normalizzazione dei pixel nell'intervallo [0.0, 1.0]
    #pragma omp parallel for
    for (int i = 0; i < num_pixels; i++) {
        result->values[i] = (float)raw_pixels[i] / 255.0f;
    }

    free(raw_pixels);
    return ERR_SUCCESS;
}

// Scrive un tensore numerico 2D su file salvandolo come immagine PGM binaria (P5)
int tensor_write_pgm(tensor* string_t, tensor* numeric_t) {
    if ((string_t  ==  NULL)  ||  (numeric_t  ==  NULL)) {
        return ERR_NULL_PTR;
    }

    ASSERT_STRING(string_t);
    ASSERT_NUMERIC(numeric_t);

    // Verifica che il tensore da esportare sia effettivamente una matrice (2D)
    int matrix_result = is_matrix(numeric_t);
    if (matrix_result != ERR_SUCCESS) return matrix_result;

    FILE* out_file = fopen(string_t->filename, "wb");
    if (out_file  ==  NULL) {
        return ERR_FILE_OPEN;
    }

    // Scrittura dell'intestazione PGM standard (magic number, dimensioni e valore massimo)
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n", numeric_t->shape[1], numeric_t->shape[0]);
    fprintf(out_file, "255\n");

    int num_pixels = numeric_t->shape[0] * numeric_t->shape[1];
    unsigned char* pixels = (unsigned char*) malloc(num_pixels * sizeof(unsigned char));
    
    if (pixels == NULL) {
        fclose(out_file);
        return ERR_OUT_OF_MEMORY;
    }

    // Conversione dei valori float in pixel a 8-bit [0, 255]
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

// Legge un tensore da file sfruttando la memoria mappata (mmap) senza effettuare copie in RAM
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

    // Acquisizione delle informazioni sulla dimensione del file tramite fstat
    struct stat file_stat;
    if (fstat(fd, &file_stat) < 0) {
        close(fd);
        free_tensor(t);
        return ERR_FILE_STAT;
    }

    // Mappatura del file direttamente nello spazio di indirizzamento virtuale
    float* mapped_data = mmap(NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_data  ==  MAP_FAILED) {
        fclose(out_file);
        free_tensor(t);
        return ERR_MMAP_FAILED;
    }

    // Suggerimento al kernel riguardo alle future letture sequenziali delle pagine di memoria
    madvise(mapped_data, file_stat.st_size, MADV_WILLNEED);
    
    fclose(out_file);

    // Estrazione dell'intestazione posizionata all'inizio della regione mappata
    on_disk_tensor* header = (on_disk_tensor*) mapped_data;

    result->type = TYPE_NUMERIC_MMAP;
    result->ndim = header->ndim;
    
    // Configurazione delle dimensioni del tensore in base al numero di dimensioni registrato
    if (header->ndim == 1) {
        result->shape[0] = header->shape[0];
        result->shape[1] = 0;
    } else {
        result->shape[0] = header->shape[0];
        result->shape[1] = header->shape[1];
    }

    // Puntatore diretto ai dati effettivi situati dopo l'offset di allineamento
    result->values = (float*) ((char*)mapped_data + header->data_offset);
    
    // Allocazione e inizializzazione del contatore di riferimenti condiviso
    result->ref_count = (int*) malloc(sizeof(int));
    if (result->ref_count == NULL) {
        munmap(mapped_data, file_stat.st_size);
        free_tensor(t);
        return ERR_OUT_OF_MEMORY;
    }
    *(result->ref_count) = 1;

    return ERR_SUCCESS;
}


// Salva un tensore su file in formato binario strutturato con intestazione e allineamento a 64 byte
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

    // Popolamento dei metadati dell'intestazione in base alla dimensionalità del tensore
    if (numeric_t->ndim == 1) {
        header.ndim = 1;
        header.shape[0] = numeric_t->shape[0];
        header.shape[1] = 0;
    } else {
        header.ndim = 2;
        header.shape[0] = numeric_t->shape[0];
        header.shape[1] = numeric_t->shape[1];
    }

    // Impostazione dell'offset fisso a 64 byte per garantire il corretto allineamento dei dati
    header.data_offset = 64;

    // Scrittura dell'intestazione sul file
    size_t written_header = fwrite(&header, sizeof(on_disk_tensor), 1, out_file);
    if (written_header != 1) {
        fclose(out_file);
        return ERR_IO;
    }

    // Inserimento dei byte di padding necessari per raggiungere esattamente l'offset di 64 byte
    int padding_size = 64 - sizeof(on_disk_tensor);
    if (padding_size > 0) {
        char padding[64] = {0};
        fwrite(padding, sizeof(char), padding_size, out_file);
    }
    
    // Calcolo del numero totale di elementi da scrivere nel flusso binario
    int num_elements = numeric_t->shape[0];
    if (numeric_t->ndim == 2) {
        num_elements *= numeric_t->shape[1];
    }

    // Scrittura sequenziale dei dati numerici in virgola mobile
    size_t written_data = fwrite(numeric_t->values, sizeof(float), num_elements, out_file);
    
    if ((int) written_data != num_elements) {
        fclose(out_file);
        return ERR_IO;
    }

    fclose(out_file);

    return ERR_SUCCESS;
}