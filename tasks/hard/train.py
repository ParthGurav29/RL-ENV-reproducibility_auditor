import argparse
import yaml
import os
import torch
from dataset import HardDataset
from model import ComplexModel

# BUG: PYTHONHASHSEED is not set, causing dicts and sets to iterate randomly
# BUG: CUBLAS_WORKSPACE_CONFIG is not set, so cuBLAS may choose non-deterministic algorithms

def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    # BUG: CLI args override config seeds without re-seeding all libraries
    parser.add_argument("--override_seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # The override logic introduces instability
    active_seed = args.override_seed if args.override_seed else config['experiment']['seed']
    set_seeds(active_seed)

    print(f"Starting training with seed: {active_seed}")
    
    dataset = HardDataset()
    model = ComplexModel()
    
    # ... dummy training logic would go here ...
    print("Training finished.")

if __name__ == "__main__":
    main()