import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def get_size(path: Path) -> int:
    """
    Calculate the total size (in bytes) of a directory or file.

    Args:
        path (Path): The file or directory path to calculate size for.

    Returns:
        int: Total size in bytes. Returns 0 if the path does not exist.
    """
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human_readable(size: int) -> str:
    """
    Convert a size in bytes to a human-readable format.

    Args:
        size (int): Size in bytes.

    Returns:
        str: Size formatted in a human-readable string (e.g., KB, MB, GB).
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def print_cache_info(name: str, path: Path):
    """
    Print cache information for a given directory path.

    Args:
        name (str): The label/name of the cache.
        path (Path): The directory path of the cache.
    """
    print("=" * 80)
    print(f"{name} cache: {path}")
    if not path.exists():
        print("Warning: This path does not exist.")
        return

    size = get_size(path)
    print(f"Total size: {human_readable(size)}")
    print("Subdirectories:")
    for p in sorted([p for p in path.iterdir() if p.is_dir()]):
        print(f"  - {p.name}")


def main():
    """
    Load environment variables and print cache information for HuggingFace and Torch.
    """
    load_dotenv()
    hf_home = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser()
    torch_home = Path(os.getenv("TORCH_HOME", "~/.cache/torch")).expanduser()

    print_cache_info("HuggingFace (HF_HOME)", hf_home)
    print_cache_info("Torch (TORCH_HOME)", torch_home)


if __name__ == "__main__":
    sys.exit(main())
