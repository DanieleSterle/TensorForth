# Daniele Sterle SM3201594

# Makefile progettato in modo tale da:
# - cercare i file header nella cartella include
# - creare i file oggetto nella cartella build
# - generare il file eseguibile nella cartella principale (Tensorforth)

CC = gcc
CFLAGS = -Wall -Wextra -O3 -I ./include -fopenmp -lm
LDFLAGS = -fopenmp -lm

# Definizione delle directory di progetto
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

# Nome del file eseguibile finale
TARGET = tensorforth

# Risoluzione automatica dei file sorgente e dei relativi oggetti
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))
OBJS += $(BUILD_DIR)/main.o

# Regola principale per la creazione dell'eseguibile
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

# Regola di compilazione per i file sorgente presenti nella cartella src/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Regola di compilazione specifica per il file main.c nella cartella principale
$(BUILD_DIR)/main.o: main.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Creazione automatica della directory di build, se non esistente
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Pulizia dei file oggetto e dell'eseguibile generato
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean