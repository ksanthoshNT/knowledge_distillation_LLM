from knowledge_distillation.src.main.evaluation.eval_args import EvalArguments
from knowledge_distillation.src.main.evaluation.evaluator import TransformerEvaluator


def main():
    # 1. Using config.ini file
    evaluator1 = TransformerEvaluator("config.ini")
    evaluator1.evaluate(split_name='validation')
    exit()

    # 2. Using EvalArguments
    args = EvalArguments(
        teacher_model_name="meta-llama/Llama-2-7b-chat-hf",
        student_model_path="./distilled_model",
        dataset_name="lamini/spider_text_to_sql"
    )
    evaluator2 = TransformerEvaluator(args)

    # 3. Using dictionary
    config_dict = {
        "teacher_model_name": "meta-llama/Llama-2-7b-chat-hf",
        "student_model_path": "./distilled_model",
        "dataset_name": "lamini/spider_text_to_sql"
    }
    evaluator3 = TransformerEvaluator(config_dict)

    # 4. Using defaults
    evaluator4 = TransformerEvaluator()

    # Run evaluation with any of the above
    evaluator1.evaluate()


if __name__ == "__main__":
    main()