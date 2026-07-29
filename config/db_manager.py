import sqlite3
import os
import yaml


class ConfigDB:
    def __init__(self, db_path="giacobazzi.db", yaml_path="config\\config_base.yaml",
                 profili_path="config\\cap_profile.yaml"):
        self.db_path = db_path
        self.yaml_path = yaml_path
        self.profili_path = profili_path
        self._inizializza_db()

    def _inizializza_db(self):
        """Creo DB e inseirisoc di valori di default se non esiste"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS configurazione(
                    parametro TEXT PRIMARY KEY,
                    valore TEXT,
                    descrizione TEXT
                    )
            ''')

            # ----NEW--- inserisco pareametri da file yamls
            if os.path.exists(self.yaml_path):
                try:
                    with open(self.yaml_path, 'r', encoding='utf-8') as file:
                        dati_yaml = yaml.safe_load(file)

                    valori_iniziali = []
                    for parametro, info in dati_yaml.items():
                        valori_iniziali.append((parametro, str(info['valore']), info['descrizione']))

                    cursor.executemany("INSERT OR IGNORE INTO configurazione VALUES (?, ?, ?)", valori_iniziali)
                except Exception as e:
                    print(f"[ERRORE] Impossibile elaborare {self.yaml_path}: {e}")
            else:
                print(f"[WARNING] File '{self.yaml_path}' non trovato. I default non verrannò caricati.")

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"[ATTENZIONE] Errore in inizializzazione DB ({e}). Il sistema proverà il fallback su yaml.")

    def _leggi_da_yaml(self):
        raw_param = {}
        if os.path.exists(self.yaml_path):
            try:
                with open(self.yaml_path, 'r', encoding='utf-8') as file:
                    dati_yaml = yaml.safe_load(file)
                    for k, v in dati_yaml.items():
                        raw_param[k] = v['valore']
                print("[SISTEMA] Fallback da yaml attivato: Dati caricati con successo.")
            except Exception as e:
                print(f"[ERRORE CRITICO] Impossibile leggere anche lo yaml ({e}).")
        else:
            print(f"[ERRORE CRITICO] file yaml '{self.yaml_path}' non trovato")

        return raw_param

    def get_param(self):
        """Recupero e formatto i parametri GLOBALI (validi per qualunque tappo).

        I parametri specifici del singolo tipo di tappo (modello, ROI,
        metodo di rotazione, soglia...) NON sono qui: si ottengono con
        get_tappo_profile(), dopo aver letto TAPPO_TYPE da questo dict.
        """
        raw_param = {}

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT parametro, valore FROM configurazione")
            raw_param = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
        except Exception as e:
            print(f"[ERRORE DB] Database inaccessile ({e}). Innesco proceudura recupro yaml... ")
            raw_param = self._leggi_da_yaml()

        cfg = {
            "BUFFER_DIR": raw_param.get("BUFFER_DIR", "buffer_telecamera"),
            "OUTPUT_DIR": raw_param.get("OUTPUT_DIR", "risultati_linea"),
            "DEVICE": raw_param.get("DEVICE", "cpu"),

            "TAPPO_TYPE": raw_param.get("TAPPO_TYPE", "tappo_rosso"),

            "OVERLAY_ORIGINALE": float(raw_param.get("OVERLAY_ORIGINALE", 0.7)),
            "OVERLAY_HEATMAP": float(raw_param.get("OVERLAY_HEATMAP", 0.3)),
            "PAUSA_BUFFER_VUOTO": float(raw_param.get("PAUSA_BUFFER_VUOTO", 0.05)),
        }
        return cfg

    def get_tappo_profile(self, tappo_type: str) -> dict:
        """
        Carica il profilo di configurazione per un dato tipo di tappo da
        config/tappi_profili.yaml (MODEL_PATH, ROI, metodo di rotazione,
        parametri specifici del metodo, soglia di rigetto, ecc.).

        Solleva ValueError con un messaggio chiaro se il file non esiste,
        se `tappo_type` non è tra i profili disponibili, se mancano chiavi
        essenziali, o se la ROI definita non è geometricamente valida —
        meglio fermare subito il processo con un errore leggibile che
        partire con una configurazione sbagliata in produzione.
        """
        if not os.path.exists(self.profili_path):
            raise ValueError(
                f"[ERRORE CRITICO] File profili tappi '{self.profili_path}' non trovato."
            )

        with open(self.profili_path, 'r', encoding='utf-8') as file:
            profili = yaml.safe_load(file) or {}

        if tappo_type not in profili:
            disponibili = ", ".join(sorted(profili.keys()))
            raise ValueError(
                f"[ERRORE CRITICO] TAPPO_TYPE='{tappo_type}' non trovato in "
                f"'{self.profili_path}'. Profili disponibili: {disponibili}"
            )

        profilo = profili[tappo_type]

        # Normalizzazione minima: assicura che le chiavi essenziali esistano,
        # con errori chiari invece di KeyError criptici più avanti nel codice.
        richieste = ["MODEL_PATH", "ROTATION_METHOD", "TAPPO_Y1", "TAPPO_Y2",
                     "TAPPO_X1", "TAPPO_X2", "SOGLIA_TAPPO", "PERCENTILE_TAPPO"]
        mancanti = [k for k in richieste if k not in profilo]
        if mancanti:
            raise ValueError(
                f"[ERRORE CRITICO] Profilo '{tappo_type}' incompleto in "
                f"'{self.profili_path}': mancano le chiavi {mancanti}"
            )

        # Validazione geometrica della ROI: le sole chiavi presenti non
        # bastano, i valori devono anche avere senso (y2>y1, x2>x1,
        # nessun valore negativo). Un profilo con placeholder a 0/0/0/0
        # passerebbe la sola verifica di presenza-chiave e poi produrrebbe
        # uno slice vuoto -> crash di np.percentile al primo frame elaborato
        # con quel profilo (vedi "ROI non validata" nella review).
        try:
            y1 = int(profilo["TAPPO_Y1"])
            y2 = int(profilo["TAPPO_Y2"])
            x1 = int(profilo["TAPPO_X1"])
            x2 = int(profilo["TAPPO_X2"])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"[ERRORE CRITICO] Profilo '{tappo_type}' in '{self.profili_path}': "
                f"coordinate ROI non numeriche (TAPPO_Y1/Y2/X1/X2): {e}"
            )

        problemi_roi = []
        if y1 < 0 or x1 < 0:
            problemi_roi.append(f"TAPPO_Y1/X1 negativi (Y1={y1}, X1={x1})")
        if y2 <= y1:
            problemi_roi.append(f"TAPPO_Y2 ({y2}) deve essere maggiore di TAPPO_Y1 ({y1})")
        if x2 <= x1:
            problemi_roi.append(f"TAPPO_X2 ({x2}) deve essere maggiore di TAPPO_X1 ({x1})")

        if problemi_roi:
            raise ValueError(
                f"[ERRORE CRITICO] Profilo '{tappo_type}' in '{self.profili_path}' "
                f"ha una ROI non valida: {'; '.join(problemi_roi)}"
            )

        profilo.setdefault("ROTATION_PARAMS", {})
        profilo.setdefault("DEVICE", None)  # None -> il main userà il DEVICE globale

        return profilo


if __name__ == "__main__":
    pass