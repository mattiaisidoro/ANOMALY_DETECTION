# qui genero tutti e tre i tpi di modelli per valutare poi performance in inferenza
# TORCH, VINO 32 E VINO 16
# USARE INT8  non migliora prestazioni , gia testato

#MODELLO ANOMALIB FA RESIZE AUTOMATICA A 256, 256 PER ESSRE IN LINEA CON LA BACKBONE PREADDESTRATA APPUNTO SU IMMAGIN I 256X256

import os
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import Folder
from anomalib.deploy import ExportType,  CompressionType

import pytorch_lightning as pl

DATASET_PATH = "./Dataset"
#DATASET_PATH = "./dataset_tappi_rossi"
#DATASET_PATH = "./dataset_trasparente_piccolo"




def main():
    print("Avvio ispezione Pathcore")
    pl.seed_everything(42, workers=True)



    datamodule = Folder(
        name= "tappi",
        root=DATASET_PATH,
        normal_dir= "train/good",
        abnormal_dir= "test/abnormal",
        normal_test_dir= "test/normal",
        seed=42,
       )

    #ora inizializzo modello
    model = Patchcore (
       
        backbone="wide_resnet50_2",
        pre_trained=True,
        coreset_sampling_ratio= 0.1,
        #post_processor=False,
    )
    engine = Engine(
        default_root_dir="./risultati/full_set" # dove salvi risultati del train
    )
    
    #qui inizio training, cioe estrazioni feature
    print("Inizio estrazioni feature patch...")
    engine.fit(datamodule=datamodule, model=model)

    #esporto torch
    percorso_torch = "./risultati/full_set/torch"
    engine.export(
        model=model,
        export_type= ExportType.TORCH,
        export_root= percorso_torch,
        datamodule=datamodule
    )

    percorso_int_8= "./risultati/full_set\openvino_8"
    engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=percorso_int_8,
        datamodule=datamodule,
        compression_type=CompressionType.INT8_PTQ,
        
    )

    percorso_vino_32 = "./risultati/full_set/openvino_32"
    engine.export(
        model=model,
        export_type= ExportType.OPENVINO,
        export_root= percorso_vino_32,
        datamodule=datamodule
    )
    
    model.model.half()
    percorso_vino_16 = "./risultati/full_set/openvino_16"
    engine.export(
        model=model,
        export_type= ExportType.OPENVINO,
        export_root= percorso_vino_16,
        datamodule=datamodule
    )
   
   

    print("Addestramento completo")

if __name__ == '__main__':
    main()


