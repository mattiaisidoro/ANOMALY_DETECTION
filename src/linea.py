#SIMULO COMPORTAMENTO TELECANERA SU LINEA PLC, CON 1 VACSHETTA OGNI 0.5 SEC

import os
import time
import shutil
import glob 

def main():
    print("SIMULO TELECAMERA")

    sorgente_dir = "da_elaborare"
    buffer_dir = "buffer_telecamera"

    os.makedirs(buffer_dir, exist_ok=True)

    lista_foto = glob.glob(os.path.join(sorgente_dir, "*.bmp")) #estraggo tutte le foto da carella
    
    if not lista_foto:
        print("NESSUNA IMMAGINE TROVATA")
        return
    
    print(f"Trovatre {len(lista_foto)}, avvio simulazione nastro in 3 secondi...")
    time.sleep(3)

    for indice, foto_originale in enumerate(lista_foto, start=1):
        nome_file = os.path.basename(foto_originale)
        destinazione = os.path.join(buffer_dir, nome_file)


        #simulo scatto
        shutil.copy(foto_originale, destinazione)
        print(f"SCATTO [CAM] {indice}/{len(lista_foto)}: {nome_file} inviato al buffer")

        #simulo attesa nastro di 0.5 secondi
        time.sleep(0.5)
       

    print("FOTO ESAURITE")

if __name__ == "__main__":
    main()