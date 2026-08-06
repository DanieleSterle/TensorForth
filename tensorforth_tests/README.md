# TensorForth — Suite di 275 test

Questa cartella contiene 275 programmi TensorForth (`tests/test_001_*.tf` …
`tests/test_275_*.tf`) pensati per esercitare il vostro interprete su:

- sintassi dei tensori (spazi, virgole, parentesi non bilanciate, casi limite)
- tutte le operazioni aritmetiche, di comparazione, logiche, di selezione
- operazioni specifiche sui tensori (`@`, `.`, `c`)
- operazioni sulla forma (`r`, `_`, `#`) inclusi i limiti di `MAX_DIM`
- generazione casuale (`?`)
- operazioni elemento per elemento e di riduzione (`R`, `m`, `M`, `S`)
- `f` (fill) e `p` (print)
- manipolazione dello stack (`d`, `s`, `o`, `D`)
- I/O su file PGM e sul formato binario custom (`(`, `)`, `{`, `}`)
- stringhe/filename ed edge case di parsing
- i tre programmi di esempio riportati nel testo del progetto, più alcuni
  programmi composti più complessi

La seconda batch (test 101-175, generata da `generate_combined.py`) è
dedicata interamente a **catene di operazioni combinate** (mai un solo
operatore isolato), suddivisa in tre gruppi da 25:

- **101-125**: catene che devono avere **successo** end-to-end
- **126-150**: catene che iniziano bene ma **falliscono** ad un certo
  punto — utili per verificare che l'errore emerga esattamente dove
  atteso e che le operazioni riuscite prima dell'errore non lascino
  risorse non liberate
- **151-175**: **casi limite** (aliasing da `d`, reshape a shape zero o
  negativa, maschere tutte-0/tutte-1, catene lunghe di dup/drop, ecc.),
  molti pensati apposta per stressare stack e reference counting — ottimi
  da rilanciare con `-v` (valgrind)

La terza batch (test 176-225, generata da `generate_extreme.py`) è
**EXTREME**: solo casi limite, nessuno con un unico output "corretto"
garantito dal testo. Copre parsing numerico ambiguo (`.5`, `5.`, `+5`,
notazione scientifica), overflow/underflow in floating point, shape enormi
o negative, stress dello stack (500 push senza pop, 50 dup/drop, catene di
`over`/`swap` ripetute), alias e mutazione in-place via `dup`, file con
nomi Unicode o vuoti, whitespace estremo (tab, CRLF, righe vuote multiple),
e un "final boss" finale che combina tutti gli operatori in un solo
programma. L'obiettivo qui non è un output esatto ma: **nessun crash, mai**,
comportamento coerente sui casi ambigui, tempi ragionevoli sui casi di
stress.

La quarta batch (test 226-275, generata da `generate_io.py`) è dedicata
esclusivamente alle **operazioni di I/O** (`(`, `)`, `{`, `}`) — **nessuno
di questi script usa l'operatore `p`**. Sono pipeline realistiche
(soglia bianco/nero, rilevamento bordi con Sobel/Laplaciano, sfocatura,
unsharp masking, round trip PGM↔binario) che scrivono sempre il risultato
su file. La correttezza va verificata ispezionando i file prodotti su
disco o l'exit code, non l'output a schermo. Solo i test che leggono
`examples/cray-2.pgm` dipendono da quell'immagine di esempio; tutti gli
altri creano da soli sia gli input che gli output.

Il file **`MANIFEST.md`** è il riferimento principale: per ciascun test
riporta categoria, descrizione, e cosa l'interprete dovrebbe fare
(`OK` = esecuzione corretta con risultato atteso, `ERROR` = errore gestito
con `exit code != 0` e senza crash, `EDGE` = caso limite non specificato
univocamente dal testo, da gestire in modo ragionevole e documentato).

## Nota importante sui test di I/O

Alcuni test (categoria "I/O" ed "Esempi/completi") fanno riferimento a
`examples/cray-2.pgm`, che è uno dei file di esempio forniti con il
progetto: copiateli nella directory da cui lanciate i test, oppure
adattate i percorsi nei relativi `.tf`. I test che leggono file
inesistenti (es. `nonexistent_file_xyz.pgm`) sono invece pensati per
fallire e non richiedono nulla.

## Uso rapido

```bash
chmod +x run_all.sh
./run_all.sh /path/al/vostro/tensorforth
```

Lo script esegue ogni test e stampa l'exit code: confrontate manualmente
output ed exit code con quanto descritto in `MANIFEST.md` (non è un
oracolo automatico pass/fail, perché molti risultati — es. numeri
casuali, formattazione esatta di `p` — dipendono dalla vostra
implementazione).

## Uso con valgrind (memory leak / memory errors)

Il testo del progetto richiede esplicitamente che "il programma deve
correttamente rilasciare le risorse allocate": per verificarlo su tutti i
100 test in un colpo solo, lanciate:

```bash
./run_all.sh /path/al/vostro/tensorforth -v
```

Ogni test viene eseguito sotto:

```
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes
```

Il log completo di ciascun test viene salvato in `logs/<nome_test>.valgrind.log`,
e alla fine dell'esecuzione lo script stampa un riepilogo con:

- i test in cui valgrind ha rilevato **errori di memoria** (invalid
  read/write, uso di memoria non inizializzata, ecc.)
- i test in cui valgrind ha rilevato **memory leak veri** (`definitely
  lost` / `indirectly lost` > 0 byte — i blocchi `still reachable` a fine
  programma, tipici di allocazioni globali liberate solo all'uscita, non
  vengono conteggiati come leak da questo riepilogo, ma potete comunque
  ispezionarli nei log)

Richiede `valgrind` installato (`sudo apt install valgrind`). L'esecuzione
sotto valgrind è molto più lenta della normale: aspettatevi qualche minuto
per l'intera suite, specialmente sui test con matrici/kernel più grandi o
numeri casuali.

## Rigenerare i test

`generate.py` contiene la definizione dei primi 100 test (operatori "in
isolamento"), `generate_combined.py` i 75 test a operazioni combinate
(101-175), `generate_extreme.py` i 50 test EXTREME (176-225),
`generate_io.py` i 50 test solo I/O senza `p` (226-275). Rilanciateli in
questo ordine per rigenerare tutto in modo consistente (ognuno appende a
`MANIFEST.md`, quindi il primo va sempre eseguito per primo):

```bash
python3 generate.py
python3 generate_combined.py
python3 generate_extreme.py
python3 generate_io.py
```
