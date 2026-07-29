# Pipeline Ispezione Tappi — Anomaly Detection Real-Time

Sistema di ispezione visiva automatica per tappi su linea di produzione, basato su modelli di **anomaly detection (PatchCore)**. Riceve i frame da una telecamera (opto o simulata), li allinea, li analizza con un modello di deep learning, e classifica ogni pezzo come **GOOD** o **REJECT**, salvando un'immagine annotata con la heatmap delle anomalie.

---

## Come funziona

1. **Acquisizione**: una telecamera (o, in test, il simulatore `linea.py`) scrive i frame `.bmp` in una cartella di buffer.
2. **Allineamento**: ogni frame viene ruotato/allineato secondo il metodo configurato per il tipo di tappo attivo (rilevamento fessura su scala di grigi, oppure template matching su marker simmetrici).
3. **Inferenza**: il frame allineato viene passato a un modello PatchCore (backend Torch o OpenVINO, rilevato automaticamente dall'estensione del file modello).
4. **Scoring**: viene calcolato un punteggio di anomalia sulla regione di interesse (ROI) del tappo, tramite percentile sulla heatmap grezza.
5. **Classificazione**: il punteggio viene confrontato con una soglia configurabile → `GOOD` o `REJECT`.
6. **Output**: viene salvata un'immagine annotata (frame + overlay heatmap sulla ROI). Il frame di input viene poi rimosso dal buffer.

Tutta la configurazione specifica per tipo di tappo (modello, ROI, metodo di rotazione, soglia) vive in un profilo YAML separato — aggiungere un nuovo tipo di tappo non richiede modifiche al codice.

---

## Struttura del progetto

```
.
├── main.py                      # entry point: loop di ispezione real-time
├── config/
│   ├── config_base.yaml         # parametri globali di default (seed iniziale del DB)
│   ├── cap_profile.yaml         # profili per tipo di tappo (modello, ROI, rotazione, soglia)
│   └── db_manager.py            # gestione configurazione (DB SQLite + fallback YAML)
├── src/
│   ├── rotation_registry.py     # registry dei metodi di allineamento disponibili
│   ├── slot_detection.py        # allineamento tramite rilevamento fessura (scala di grigi)
│   ├── alignment.py             # allineamento tramite template matching su marker simmetrici
│   └── linea.py                 # SOLO PER TEST: simula l'arrivo dei frame da telecamera/nastro
├── buffer_telecamera/           # cartella di ingresso frame (creata automaticamente)
│   └── quarantena/              # frame che hanno causato un errore di elaborazione
├── risultati_linea/             # immagini annotate in output (creata automaticamente)
├── logs/                        # log applicativi con rotazione (creata automaticamente)
├── giacobazzi.db                # DB SQLite di configurazione (NON versionato in git)
└── CODE_REVIEW_STATUS.md        # stato della code review: cosa è stato sistemato e cosa resta
```

---

## Configurazione

La configurazione è divisa in due livelli:

### 1. Parametri globali (validi per qualunque tipo di tappo)

Gestiti da `ConfigDB.get_param()`, letti dal database SQLite (`giacobazzi.db`). Il DB viene seminato la prima volta dai valori in `config/config_base.yaml`; dagli avvii successivi, **il DB è la fonte di verità** — modifiche fatte direttamente sul DB restano valide, lo YAML non le sovrascrive (workflow pensato per essere gestito da personale tecnico/assistenza, non dal cliente finale).

| Parametro | Tipo | Descrizione |
|---|---|---|
| `BUFFER_DIR` | str | Cartella monitorata per i frame in arrivo |
| `OUTPUT_DIR` | str | Cartella dove salvare le immagini annotate |
| `DEVICE` | str | Device di calcolo di default: `"cpu"`, `"cuda"`, `"cuda:<indice>"` |
| `TAPPO_TYPE` | str | Tipo di tappo attivo; seleziona il profilo da `cap_profile.yaml` |
| `PAUSA_BUFFER_VUOTO` | float | Secondi di attesa quando il buffer è vuoto |
| `OVERLAY_ORIGINALE` / `OVERLAY_HEATMAP` | float | Pesi di blending per l'overlay heatmap sull'immagine originale |

### 2. Parametri per tipo di tappo

Gestiti da `ConfigDB.get_tappo_profile(tappo_type)`, letti da `config/cap_profile.yaml`.

| Parametro | Tipo | Descrizione |
|---|---|---|
| `MODEL_PATH` | str | Path al modello (`.pt`/`.pth` → Torch, `.xml` → OpenVINO) |
| `DEVICE` | str | Sovrascrive il `DEVICE` globale, se specificato |
| `ROTATION_METHOD` | str | Chiave nel registry (`src/rotation_registry.py`) del metodo di allineamento |
| `ROTATION_PARAMS` | dict | Parametri passati al metodo di rotazione scelto |
| `TAPPO_Y1/Y2/X1/X2` | int | Bounding box in pixel della ROI del tappo (validati: `Y2>Y1`, `X2>X1`, nessun negativo) |
| `SOGLIA_TAPPO` | float | Soglia del punteggio di anomalia oltre la quale un pezzo è `REJECT` |
| `PERCENTILE_TAPPO` | float | Percentile usato per calcolare il punteggio sulla ROI |

**Aggiungere un nuovo tipo di tappo:** basta aggiungere un blocco in `cap_profile.yaml` con i parametri sopra. Se serve un nuovo algoritmo di allineamento, va aggiunto al registry in `src/rotation_registry.py`. `main.py` non richiede modifiche.

**Cambiare tipo di tappo attivo:** il profilo viene letto una sola volta all'avvio; per cambiare `TAPPO_TYPE` è necessario riavviare il processo (comportamento voluto: si riavvia ad ogni cambio lotto/linea).

---

## Avvio

```bash
python main.py
```

> Nota: se l'entry point viene lanciato da una posizione diversa dalla root del progetto, o se il layout dei moduli richiede l'esecuzione come pacchetto, usare:
> ```bash
> python -m src.main
> ```

Il processo, una volta avviato:
1. Carica configurazione globale e profilo del tappo attivo.
2. Carica il modello di anomaly detection (Torch o OpenVINO, backend rilevato automaticamente).
3. Entra in un loop continuo: attende frame in `BUFFER_DIR`, li elabora, salva l'output, rimuove l'input.

---

## Logging

I log vengono scritti in `logs/inspection.log`, con rotazione automatica (10 MB per file, ultime 5 rotazioni conservate — max ~50 MB totali). In console vengono mostrati i messaggi di livello `INFO` e superiore (parametri caricati, profilo attivo, verdetto di ogni immagine con punteggio e latenza); nel file di log sono presenti anche i dettagli di livello `DEBUG`, utili per il troubleshooting.

---

## Policy di salvataggio output

I frame classificati `REJECT` vengono **sempre** salvati in `OUTPUT_DIR` (servono per tarare la soglia, analizzare i difetti, rispondere a contestazioni).

I frame `GOOD` sono la stragrande maggioranza del volume prodotto e, in produzione, in genere non servono a nessuno: il loro salvataggio è controllato dal flag `SALVA_ANCHE_GOOD` in cima a `main.py`.

- **Produzione** (default): `SALVA_ANCHE_GOOD = False` → solo i `REJECT` vengono salvati.
- **Demo/collaudo**: decommentare la riga di override `SALVA_ANCHE_GOOD = True` per salvare anche i `GOOD`.

---

## Gestione degli errori

- Un frame che causa un'eccezione durante l'elaborazione (allineamento, inferenza, scrittura output, ecc.) **non blocca la linea**: viene spostato in `buffer_telecamera/quarantena/` per analisi successiva, e il processo continua con il frame successivo.
- Se il salvataggio dell'immagine di output fallisce (es. disco pieno), il frame di input **non viene cancellato** e finisce anch'esso in quarantena, per non perdere traccia del verdetto.

---

## Formato immagini

Il formato usato in tutta la pipeline (buffer di ingresso e output annotato) è **BMP** non compresso: nessun artefatto di compressione lossy che potrebbe interferire con il punteggio di anomalia, lettura/scrittura più leggera per il loop real-time. Sarà anche il formato nativo prodotto dalla telecamera opto in produzione.

---

## Requisiti

```
opencv-python
numpy
torch
anomalib
openvino
pyyaml
```

> Nota: l'elenco completo delle dipendenze presente attualmente in `pyproject.toml`/`requirements.txt` è più ampio di quanto il progetto usi realmente — è in programma una pulizia (vedi `CODE_REVIEW_STATUS.md`).

---

## Stato della code review

Per lo storico dei problemi individuati dalla code review, cosa è stato risolto, cosa è stato discusso e archiviato consapevolmente, e cosa resta da fare, vedi [`CODE_REVIEW_STATUS.md`](./CODE_REVIEW_STATUS.md).