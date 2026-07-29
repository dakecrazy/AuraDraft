"""Document loader — read text files from disk.

M1: txt / md only.  PDF, docx, images are M3+.
"""

from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".rst", ".log"})


def load_text(path: str | Path) -> str:
    """Read a text file, returning its content as a string.

    Tries utf-8 first; falls back to latin-1 if that fails so we never
    crash on a binary file (the caller should check *supported()* first).
    """
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1")


def supported(path: str | Path) -> bool:
    """Return True if *path* has a recognised text-file extension."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def iter_documents(root: str | Path, glob: str = "**/*") -> list[Path]:
    """Recursively list all supported documents under *root*."""
    root = Path(root)
    return sorted(p for p in root.glob(glob) if supported(p))