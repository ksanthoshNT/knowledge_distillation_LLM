from dataclasses import dataclass
from typing import Optional, Union
import configparser


@dataclass
class EvalArguments:
    """HuggingFace-style arguments for evaluation."""
    teacher_model_name: str
    student_model_path: str
    dataset_name: str
    device: str = "auto"
    dataset_config: Optional[str] = None
    num_samples: int = 10
    output_dir: str = "evaluation_results"

    # Generation config
    max_new_tokens: int = 100
    num_return_sequences: int = 1
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0

class TransformerConfig:
    """Unified configuration handler supporting both file and argument-based configs."""

    def __init__(self, config_source: Union[str, EvalArguments, dict, None] = None):
        """
        Initialize configuration from multiple possible sources.

        Args:
            config_source: Can be one of:
                - str: Path to config.ini file
                - EvalArguments: HuggingFace-style arguments
                - dict: Dictionary of configuration values
                - None: Use default values
        """
        if isinstance(config_source, str):
            self._load_from_file(config_source)
        elif isinstance(config_source, EvalArguments):
            self._load_from_args(config_source)
        elif isinstance(config_source, dict):
            self._load_from_dict(config_source)
        else:
            self._load_defaults()

    def _load_from_file(self, config_path: str):
        """Load configuration from ini file."""
        config = configparser.ConfigParser()
        config.read(config_path)

        print(config.sections())

        self.model_config = {
            'teacher_model_name': config['models']['teacher_model_name'],
            'student_model_path': config['models']['student_model_path'],
            'device': config['models']['device']
        }

        self.dataset_config = {
            'name': config['dataset']['name'],
            'config': None if config['dataset']['config'] == 'null' else config['dataset']['config'],
            'num_samples': config['dataset'].getint('num_samples'),
            'output_dir': config['dataset']['output_dir']
        }

        self.generation_config = {
            'max_new_tokens': config['generation'].getint('max_new_tokens'),
            'num_return_sequences': config['generation'].getint('num_return_sequences'),
            'do_sample': config['generation'].getboolean('do_sample'),
            'temperature': config['generation'].getfloat('temperature'),
            'top_p': config['generation'].getfloat('top_p')
        }

    def _load_from_args(self, args: EvalArguments):
        """Load configuration from EvalArguments."""
        self.model_config = {
            'teacher_model_name': args.teacher_model_name,
            'student_model_path': args.student_model_path,
            'device': args.device
        }

        self.dataset_config = {
            'name': args.dataset_name,
            'config': args.dataset_config,
            'num_samples': args.num_samples,
            'output_dir': args.output_dir
        }

        self.generation_config = {
            'max_new_tokens': args.max_new_tokens,
            'num_return_sequences': args.num_return_sequences,
            'do_sample': args.do_sample,
            'temperature': args.temperature,
            'top_p': args.top_p
        }

    def _load_from_dict(self, config_dict: dict):
        """Load configuration from dictionary."""
        self.model_config = {
            'teacher_model_name': config_dict.get('teacher_model_name'),
            'student_model_path': config_dict.get('student_model_path'),
            'device': config_dict.get('device', 'auto')
        }

        self.dataset_config = {
            'name': config_dict.get('dataset_name'),
            'config': config_dict.get('dataset_config'),
            'num_samples': config_dict.get('num_samples', 10),
            'output_dir': config_dict.get('output_dir', 'evaluation_results')
        }

        self.generation_config = {
            'max_new_tokens': config_dict.get('max_new_tokens', 100),
            'num_return_sequences': config_dict.get('num_return_sequences', 1),
            'do_sample': config_dict.get('do_sample', True),
            'temperature': config_dict.get('temperature', 1.0),
            'top_p': config_dict.get('top_p', 1.0)
        }

    def _load_defaults(self):
        """Load default configuration values."""
        self.model_config = {
            'teacher_model_name': "meta-llama/Llama-2-7b-chat-hf",
            'student_model_path': "./distilled_model",
            'device': "auto"
        }

        self.dataset_config = {
            'name': "lamini/spider_text_to_sql",
            'config': None,
            'num_samples': 10,
            'output_dir': "evaluation_results"
        }

        self.generation_config = {
            'max_new_tokens': 100,
            'num_return_sequences': 1,
            'do_sample': True,
            'temperature': 1.0,
            'top_p': 1.0
        }