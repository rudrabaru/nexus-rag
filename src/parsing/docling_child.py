"""
The code that runs inside the Docling child process (src/parsing/files.py starts it).

Kept in its own module so the child imports only this and Docling, and the parent never
imports Docling at all: the parent stays small, and the child's memory is returned to the
OS when it exits (Docling issues #366/#474: memory is not released between documents).
"""


def convert(path: str, conn) -> None:
    try:
        from docling.document_converter import DocumentConverter

        result = DocumentConverter().convert(path, raises_on_error=True)
        markdown = result.document.export_to_markdown(
            # & and _ stay literal: escaped forms put "amp" into the keyword index and break
            # identifiers such as snake_case names in code documentation.
            escape_html=False,
            escape_underscores=False,
            image_placeholder="",
        )
        conn.send(("ok", markdown))
    except Exception as e:  # reported to the parent, which decides on a fallback
        conn.send(("error", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()
