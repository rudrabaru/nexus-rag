"""
What runs inside the parser child process (src/parsing/files.py starts it):

    python -m src.parsing.child <mode> <input file> <output file>

Every library that reads an untrusted upload (Docling, PyMuPDF, the zip reader) runs here and
never in the worker, so a crash, a hang or an out-of-memory kill costs only this process. The
result travels back as a UTF-8 file, not a pickle: the worker never deserialises anything the
parsed file could have influenced.

Exit codes: 0 done; REJECTED the input itself is unusable (retrying cannot help); anything else
is a failure of the parser (timeout and kills are seen by the parent).
"""
import sys
import zipfile
from pathlib import Path
from typing import Callable, Dict, List

DONE, FAILED, REJECTED = 0, 1, 3

# A DOCX is a zip, and a small upload can unpack to gigabytes. 200 MiB is 10x the 20 MB upload
# cap: real documents compress 2-10x, so this refuses only archives that expand far beyond that.
# Experiment: not tuned on a corpus; raise it if legitimate documents are rejected.
MAX_UNPACKED_DOCX_BYTES = 200 * 1024 * 1024


class RejectedInput(Exception):
    """The file cannot be parsed and never will be."""


def page_count(path: str) -> str:
    import pymupdf

    try:
        with pymupdf.open(path) as pdf:
            return str(pdf.page_count)
    except Exception as e:
        raise RejectedInput(f"The PDF could not be opened: {e}") from e


def plain_text(path: str) -> str:
    import pymupdf

    with pymupdf.open(path) as pdf:
        return "\n\n".join(page.get_text() for page in pdf)


def docling_markdown(path: str) -> str:
    if Path(path).suffix.lower() == ".docx":
        check_unpacked_size(path)
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(path, raises_on_error=True)
    return result.document.export_to_markdown(
        # & and _ stay literal: escaped forms put "amp" into the keyword index and break
        # identifiers such as snake_case names in code documentation.
        escape_html=False,
        escape_underscores=False,
        image_placeholder="",
    )


def check_unpacked_size(path: str) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            unpacked = sum(entry.file_size for entry in archive.infolist())
    except zipfile.BadZipFile as e:
        raise RejectedInput("The file is not a valid DOCX document.") from e
    if unpacked > MAX_UNPACKED_DOCX_BYTES:
        raise RejectedInput(f"The document unpacks to {unpacked // 2**20} MiB, over the {MAX_UNPACKED_DOCX_BYTES // 2**20} MiB limit.")


MODES: Dict[str, Callable[[str], str]] = {"pages": page_count, "text": plain_text, "docling": docling_markdown}


def main(argv: List[str]) -> int:
    mode, source, target = argv
    try:
        result = MODES[mode](source)
    except RejectedInput as e:
        print(e, file=sys.stderr)
        return REJECTED
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return FAILED
    Path(target).write_text(result, encoding="utf-8")
    return DONE


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
