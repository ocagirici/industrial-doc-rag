"""PDF loading: a file path in, page-tagged text out.

We lean on LangChain's ``PyPDFLoader`` for the parsing, then immediately convert
to our own ``Page`` dataclass so nothing downstream depends on LangChain types.
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


@dataclass(frozen=True)
class Page:
    """Extracted text for a single PDF page.

    ``page`` is 1-indexed for human-facing citations (PyPDF reports 0-indexed).
    """

    source: str
    page: int
    text: str


def load_pdf(path: str | Path) -> list[Page]:
    """Load a PDF and return its non-empty pages as ``Page`` records.

    ``source`` is the file's basename, which is what citations display.
    """
    path = Path(path)
    loader = PyPDFLoader(str(path))
    pages: list[Page] = []
    for doc in loader.load():
        text = doc.page_content.strip()
        if not text:
            continue
        page_no = int(doc.metadata.get("page", 0)) + 1
        pages.append(Page(source=path.name, page=page_no, text=text))
    return pages
