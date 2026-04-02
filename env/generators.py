"""
Dynamic task file generators — 26 total violations across 3 tasks.
Each reset() randomly selects a subset, preventing memorisation.
"""
import random

EASY_ALL_VIOLATIONS = [
    "missing_random_seed", "missing_numpy_seed", "missing_torch_seed", "missing_cuda_seed",
    "unpinned_torch", "unpinned_numpy", "unpinned_scikit-learn", "unpinned_pandas",
    "missing_hashseed", "missing_cudnn_deterministic", "missing_cudnn_benchmark_off",
]

MEDIUM_ALL_VIOLATIONS = [
    "dataloader_shuffle_no_seed", "missing_deterministic_flag",
    "missing_cudnn_deterministic", "missing_cudnn_benchmark_off", "missing_worker_seed",
    "missing_generator_seed", "missing_default_rng_seed", "missing_dropout_seed",
]

HARD_ALL_VIOLATIONS = [
    "worker_seed_cross_file", "cublas_workspace_config", "package_version_conflict",
    "model_weight_init_seed", "config_yaml_override", "hash_randomization",
    "multiprocessing_no_seed",
]

ALL_VIOLATIONS_MAP = {
    "easy": EASY_ALL_VIOLATIONS,
    "medium": MEDIUM_ALL_VIOLATIONS,
    "hard": HARD_ALL_VIOLATIONS,
}

VIOLATION_RANGES = {
    "easy": (5, 9),
    "medium": (4, 6),
    "hard": (5, 7),
}


def select_violations(task: str, rng: random.Random) -> set[str]:
    all_v = ALL_VIOLATIONS_MAP[task]
    lo, hi = VIOLATION_RANGES[task]
    return set(rng.sample(all_v, rng.randint(lo, hi)))


def generate_files(task: str, active: set[str], rng: random.Random) -> dict[str, str]:
    files = {"easy": _gen_easy, "medium": _gen_medium, "hard": _gen_hard}[task](active, rng)
    # Add line numbers to all code files so agents can report accurate line references
    return {name: _add_line_numbers(content) for name, content in files.items()}


def _add_line_numbers(content: str) -> str:
    """Prepend line numbers to file content for accurate agent reporting."""
    lines = content.split('\n')
    # Strip trailing empty lines for cleaner numbering
    while lines and lines[-1].strip() == '':
        lines.pop()
    width = len(str(len(lines)))
    return '\n'.join(f'{i+1:>{width}}: {line}' for i, line in enumerate(lines)) + '\n'


# ── Easy ─────────────────────────────────────────────────────────────────────

def _gen_easy(active: set[str], rng: random.Random) -> dict[str, str]:
    imports = ["import torch", "import torch.nn as nn", "import numpy as np",
               "import pandas as pd", "from sklearn.model_selection import train_test_split"]
    if "missing_random_seed" not in active:
        imports.insert(0, "import random")
    if "missing_hashseed" not in active:
        imports.insert(0, "import os")

    seed_lines = []
    if "missing_hashseed" not in active:
        seed_lines.append('os.environ["PYTHONHASHSEED"] = "0"')
    if "missing_random_seed" not in active:
        seed_lines.append("random.seed(42)")
    if "missing_numpy_seed" not in active:
        seed_lines.append("np.random.seed(42)")
    if "missing_torch_seed" not in active:
        seed_lines.append("torch.manual_seed(42)")
    if "missing_cuda_seed" not in active:
        seed_lines.append("torch.cuda.manual_seed_all(42)")

    config_lines = []
    if "missing_cudnn_deterministic" not in active:
        config_lines.append("torch.backends.cudnn.deterministic = True")
    if "missing_cudnn_benchmark_off" not in active:
        config_lines.append("torch.backends.cudnn.benchmark = False")


    blocks = ["\n".join(imports) + "\n"]
    if seed_lines:
        blocks.append("\n# Reproducibility seeds\n" + "\n".join(seed_lines) + "\n")
    if config_lines:
        blocks.append("\n# Determinism configuration\n" + "\n".join(config_lines) + "\n")
    blocks.append(_EASY_BODY)

    pkgs = {"torch": "2.0.0", "numpy": "1.24.0", "scikit-learn": "1.2.2", "pandas": "2.0.0"}
    req = "\n".join(p if f"unpinned_{p}" in active else f"{p}=={v}" for p, v in pkgs.items()) + "\n"
    return {"train.py": "\n".join(blocks), "requirements.txt": req}

_EASY_BODY = '''
def load_data():
    df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    labels = np.random.randint(0, 2, 100)
    return train_test_split(df.values, labels, test_size=0.2)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

def train():
    X_train, X_test, y_train, y_test = load_data()
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
'''

# ── Medium ───────────────────────────────────────────────────────────────────

def _gen_medium(active: set[str], rng: random.Random) -> dict[str, str]:
    header = "import torch\nimport torch.nn as nn\nfrom torch.utils.data import DataLoader, TensorDataset\nimport numpy as np\n"

    fl = []
    if "missing_deterministic_flag" not in active:
        fl.append("torch.use_deterministic_algorithms(True)")
    if "missing_cudnn_deterministic" not in active:
        fl.append("torch.backends.cudnn.deterministic = True")
    if "missing_cudnn_benchmark_off" not in active:
        fl.append("torch.backends.cudnn.benchmark = False")
    flags = ("\n# Deterministic configuration\n" + "\n".join(fl) + "\n") if fl else "\n"

    if "missing_default_rng_seed" in active:
        data_line = "    rng = np.random.default_rng()\n    data = rng.standard_normal((1000, 10))"
    else:
        data_line = "    rng = np.random.default_rng(42)\n    data = rng.standard_normal((1000, 10))"

    if "missing_generator_seed" in active:
        gen_line = "    g = torch.Generator()\n    val_idx = torch.randperm(1000, generator=g)[:200]"
    else:
        gen_line = "    g = torch.Generator().manual_seed(42)\n    val_idx = torch.randperm(1000, generator=g)[:200]"

    dl_ex = []
    if "missing_worker_seed" not in active:
        dl_ex.append("        worker_init_fn=lambda w: torch.manual_seed(42 + w),")
    if "dataloader_shuffle_no_seed" not in active:
        dl_ex.append("        generator=torch.Generator().manual_seed(42),")
    dl_extra = ("\n" + "\n".join(dl_ex)) if dl_ex else ""

    dropout = "        nn.Dropout(0.5),\n" if "missing_dropout_seed" in active else ""

    body = '''
def train_model():
''' + data_line + '''
    inputs = torch.tensor(data, dtype=torch.float32)
    targets = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(inputs, targets)
''' + gen_line + '''

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,''' + dl_extra + '''
    )

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
''' + dropout + '''        nn.Linear(32, 2)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for batch_idx, (d, t) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(d)
        loss = criterion(output, t)
        loss.backward()
        optimizer.step()

    print("Training complete. But is it reproducible?")

if __name__ == "__main__":
    torch.manual_seed(42)
    train_model()
'''
    return {"train.py": header + flags + body, "requirements.txt": "torch==2.1.2\ntorchvision==0.16.2\nnumpy==1.26.3\n"}


# ── Hard ─────────────────────────────────────────────────────────────────────

def _gen_hard(active: set[str], rng: random.Random) -> dict[str, str]:
    # -- env setup at top of file --
    env_lines = []
    if "hash_randomization" not in active:
        env_lines.append('os.environ["PYTHONHASHSEED"] = "0"')
    if "cublas_workspace_config" not in active:
        env_lines.append('os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"')
    env_block = "\n".join(env_lines) + "\n" if env_lines else ""

    # -- set_seeds body --
    seeds = []
    seeds.append("    torch.manual_seed(seed)")
    if "config_yaml_override" not in active:
        seeds.append("    np.random.seed(seed)")
        seeds.append("    random.seed(seed)")
    seeds.append("    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(seed)")
    seeds_body = "\n".join(seeds)

    # -- main body --


    if "multiprocessing_no_seed" in active:
        mp_block = '''
    def preprocess(x):
        return x + np.random.randn() * 0.01

    with Pool(4) as pool:
        preprocessed = pool.map(preprocess, range(100))
'''
    else:
        mp_block = '''
    def preprocess(x):
        return x + np.random.randn() * 0.01

    def mp_init():
        np.random.seed(42)

    with Pool(4, initializer=mp_init) as pool:
        preprocessed = pool.map(preprocess, range(100))
'''

    train_main = '''
def main():
''' + '''    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--override_seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    active_seed = args.override_seed if args.override_seed else config['experiment']['seed']
    set_seeds(active_seed)

    print(f"Starting training with seed: {active_seed}")

    dataset = HardDataset()
    model = ComplexModel()
''' + mp_block + '''    print("Training finished.")

if __name__ == "__main__":
    main()
'''

    train_py = _HARD_HEADER + env_block + "\ndef set_seeds(seed):\n" + seeds_body + "\n" + train_main

    # -- dataset.py --
    ds_extra = ""
    if "worker_seed_cross_file" not in active:
        ds_extra = "\n\ndef worker_init_fn(worker_id):\n    np.random.seed(42 + worker_id)\n    random.seed(42 + worker_id)\n"
    dataset_py = _HARD_DATASET + ds_extra

    # -- model.py --
    guard = "        torch.manual_seed(42)  # Seed guard for weight init\n" if "model_weight_init_seed" not in active else ""
    model_py = _hard_model(guard)

    # -- requirements.txt --
    if "package_version_conflict" in active:
        req = "torch==2.1.2\ntorchvision==0.17.0\nnumpy==1.26.3\npyyaml\n"
    else:
        req = "torch==2.1.2\ntorchvision==0.16.2\nnumpy==1.26.3\npyyaml\n"

    config_yaml = 'experiment:\n  name: "hard_reproducibility_test"\n  epochs: 10\n  batch_size: 64\n  seed: 42\n  learning_rate: 0.001\n'

    return {"train.py": train_py, "dataset.py": dataset_py, "model.py": model_py,
            "config.yaml": config_yaml, "requirements.txt": req}

_HARD_HEADER = '''\
import argparse
import yaml
import os
import random
import numpy as np
import torch
from multiprocessing import Pool
from dataset import HardDataset
from model import ComplexModel

'''

_HARD_DATASET = '''\
import torch
from torch.utils.data import Dataset
import random
import numpy as np

class HardDataset(Dataset):
    def __init__(self, size=1000):
        self.data = np.random.randn(size, 20)
        self.labels = [random.randint(0, 1) for _ in range(size)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
'''

def _hard_model(init_guard: str) -> str:
    return '''\
import torch
import torch.nn as nn

class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self._initialize_weights()

    def _initialize_weights(self):
''' + init_guard + '''\
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)
'''
