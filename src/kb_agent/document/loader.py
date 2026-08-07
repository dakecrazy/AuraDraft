"""Document loader — read text files from disk.

M1: txt / md / rst / log.  PDF (M3) is supported via pymupdf with a
PyPDF2 fallback — both are optional imports so the package still works
without them (PDFs then raise a clear error instead of silently reading
binary garbage via the latin-1 fallback).
"""

from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".rst", ".log", ".pdf"})


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF using pymupdf, falling back to PyPDF2."""
    try:
        import pymupdf  # pymupdf 1.28+ (fitz name deprecated)
        doc = pymupdf.open(str(path))
        return "".join(str(page.get_text()) for page in doc)
    except ImportError:
        pass
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        return "".join(str(page.extract_text() or "") for page in reader.pages)
    except ImportError:
        raise RuntimeError(
            "PDF support requires pymupdf or PyPDF2. "
            "Install one: pip install pymupdf"
        )


def load_text(path: str | Path) -> str:
    """Read a file, returning its content as a string.

    PDFs are extracted via pymupdf (or PyPDF2 fallback). Text files try
    utf-8 first, then latin-1. The latin-1 fallback is intentionally NOT
    applied to PDFs — a binary PDF read as latin-1 produces garbage tokens
    that corrupt BM25 ranking and cluster centroids.
    """
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return _extract_pdf(p)
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