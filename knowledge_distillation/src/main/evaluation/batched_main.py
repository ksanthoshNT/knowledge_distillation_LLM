from pathlib import Path

from knowledge_distillation.src.main.evaluation.batched_evaluator import TransformerEvaluator


def main():
    # 1. Using config.ini file
    dir_path = Path(__file__).parent
    config_path =  str( dir_path / "config.ini")
    evaluator1 = TransformerEvaluator(config_path)
    evaluator1.evaluate(split_name='validation')

if __name__ == '__main__':
    main()