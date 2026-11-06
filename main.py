"""Punto de entrada estándar de PaddleX para check_dataset / train / evaluate / export.

Uso:
    python main.py -c configs/latin_rec_finetune.yaml -o Global.mode=check_dataset
    python main.py -c configs/latin_rec_finetune.yaml -o Global.mode=train
"""

from paddlex.engine import Engine

if __name__ == "__main__":
    engine = Engine()
    engine.run()
