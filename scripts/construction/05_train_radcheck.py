"""Fine-tune RadCheck 8B on the synthetic samples (Sec. 4.2, App. C.4). Needs a GPU (A100 40GB in the paper).

python scripts/construction/05_train_radcheck.py dataset.input_file=<synthetic.jsonl>
"""

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from radhalo.training.radcheck import train


@hydra.main(config_path="../../configs", config_name="train_radcheck", version_base=None)
def main(cfg: DictConfig) -> None:
    result = train(cfg)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    load_dotenv()
    main()
