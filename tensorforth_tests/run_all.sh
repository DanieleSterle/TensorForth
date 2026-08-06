#!/bin/bash
# Esegue tutti i test in tests/ contro il vostro binario tensorforth
# e mostra output ed exit code per ciascuno.
#
# Uso:
#   ./run_all.sh /path/to/tensorforth              # esecuzione normale
#   ./run_all.sh /path/to/tensorforth -v            # esecuzione sotto valgrind
#   ./run_all.sh /path/to/tensorforth --valgrind    # idem
#
# Con -v/--valgrind ogni test viene eseguito sotto
#   valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes
# e il log completo di ciascuno viene salvato in logs/<nome_test>.valgrind.log
# (utile per il requisito "il programma deve correttamente rilasciare le
# risorse allocate"). Alla fine viene stampato un riepilogo di quali test
# hanno leak, errori, o hanno definito comportamento non valido secondo
# valgrind.
#
# Non giudica automaticamente "giusto/sbagliato" sull'output del programma
# (molti test attendono un errore, altri un output specifico): usate
# MANIFEST.md come riferimento per interpretare ogni risultato.

set -u

BIN="${1:-./tensorforth}"
MODE="${2:-}"

if [ ! -x "$BIN" ]; then
    echo "Eseguibile non trovato o non eseguibile: $BIN"
    echo "Uso: $0 /path/to/tensorforth [-v|--valgrind]"
    exit 1
fi

USE_VALGRIND=0
if [ "$MODE" = "-v" ] || [ "$MODE" = "--valgrind" ]; then
    USE_VALGRIND=1
    if ! command -v valgrind >/dev/null 2>&1; then
        echo "valgrind richiesto da -v/--valgrind ma non trovato nel PATH."
        echo "Installatelo con: sudo apt install valgrind"
        exit 1
    fi
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tests"
LOGDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
TOTAL=0

VG_OPTS="--leak-check=full --show-leak-kinds=all --track-origins=yes --error-exitcode=99"

# Liste dei test problematici secondo valgrind (usate solo se USE_VALGRIND=1)
LEAKY_TESTS=()
ERROR_TESTS=()

if [ "$USE_VALGRIND" -eq 1 ]; then
    mkdir -p "$LOGDIR"
fi

for f in "$DIR"/test_*.tf; do
    TOTAL=$((TOTAL+1))
    name=$(basename "$f")
    echo "==================================================================="
    echo ">>> $name"

    if [ "$USE_VALGRIND" -eq 1 ]; then
        logfile="$LOGDIR/${name%.tf}.valgrind.log"
        valgrind $VG_OPTS --log-file="$logfile" "$BIN" "$f"
        code=$?
        echo "--- exit code: $code (log: $logfile) ---"

        # "definitely lost" o "indirectly lost" > 0 byte indicano memory leak vero
        if grep -qE "definitely lost: [1-9]|indirectly lost: [1-9]" "$logfile"; then
            LEAKY_TESTS+=("$name")
        fi
        # error-exitcode=99 segnala errori rilevati da valgrind (invalid read/write,
        # uso di memoria non inizializzata, ecc.), a prescindere dall'exit code del programma
        if [ "$code" -eq 99 ]; then
            ERROR_TESTS+=("$name")
        fi
    else
        "$BIN" "$f"
        code=$?
        echo "--- exit code: $code ---"
    fi
done

echo "==================================================================="
echo "Eseguiti $TOTAL test. Confrontate output/exit code con MANIFEST.md"

if [ "$USE_VALGRIND" -eq 1 ]; then
    echo ""
    echo "=================== RIEPILOGO VALGRIND ==========================="
    echo "Log completi in: $LOGDIR/"
    echo ""
    if [ "${#ERROR_TESTS[@]}" -eq 0 ]; then
        echo "Nessun errore di memoria (invalid read/write, uso di memoria non"
        echo "inizializzata, ecc.) rilevato da valgrind."
    else
        echo "Test con ERRORI DI MEMORIA rilevati da valgrind (${#ERROR_TESTS[@]}):"
        for t in "${ERROR_TESTS[@]}"; do
            echo "  - $t"
        done
    fi
    echo ""
    if [ "${#LEAKY_TESTS[@]}" -eq 0 ]; then
        echo "Nessun memory leak (definitely/indirectly lost) rilevato."
    else
        echo "Test con MEMORY LEAK rilevati (${#LEAKY_TESTS[@]}):"
        for t in "${LEAKY_TESTS[@]}"; do
            echo "  - $t"
        done
    fi
    echo "===================================================================="
fi
