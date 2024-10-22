import os
from pathlib import Path

from knowledge_distillation.src.main.evaluation.batched_evaluator import TransformerEvaluator


def main():
    # 1. Using config.ini file
    config_path = os.path.join(os.getcwd(),"knowledge_distillation/src/main/evaluation/config.ini")
    print(config_path)
    evaluator1 = TransformerEvaluator(config_path)
    evaluator1.evaluate(split_name='validation')

if __name__ == '__main__':
    main()