"""Turn uploaded bytes into clean plain text. """

import io
import logging
import re
from collections.abc import Callable

from pypdf import PasswordType, PdfReader

from app.core.exceptions import (
    EncryptedPdfError,
    InvalidFileError,
    NoExtractableTextError,
)
from app.services.file_validation import FileType

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF-"
_PDF_MAGIC_WINDOW = 1024  # the PDF spec allows junk before the header, up to 1 KiB


def _extract_txt(content: bytes) -> str:
    if b"\x00" in content:
        raise InvalidFileError("The file looks binary, not like a text file.")
    try:
        return content.decode("utf-8-sig")  # utf-8-sig also strips a BOM
    except UnicodeDecodeError as exc:
        raise InvalidFileError("Text files must be UTF-8 encoded.") from exc


def _extract_pdf(content: bytes) -> str:
    if _PDF_MAGIC not in content[:_PDF_MAGIC_WINDOW]:
        raise InvalidFileError("The file does not look like a valid PDF.")

    # pypdf can raise many exception types on malformed input, so the parsing
    # calls are wrapped broadly. Details are logged; the client gets a clean message.
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
            raise EncryptedPdfError(
                "The PDF is password-protected and cannot be processed."
            )
        pages = [page.extract_text() or "" for page in reader.pages]
    except EncryptedPdfError:
        raise
    except Exception as exc:
        logger.warning("PDF parsing failed: %s: %s", type(exc).__name__, exc)
        raise InvalidFileError("The PDF could not be read; it may be corrupted.") from exc
    return "\n\n".join(pages)


def _normalize(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)  # trailing spaces before a newline
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line
    return text.strip()


_EXTRACTORS: dict[FileType, Callable[[bytes], str]] = {
    FileType.PDF: _extract_pdf,
    FileType.TXT: _extract_txt,
}


def extract_text(content: bytes, file_type: FileType) -> str:
    text = _normalize(_EXTRACTORS[file_type](content))
    if not text:
        if file_type is FileType.PDF:
            raise NoExtractableTextError(
                "No text could be extracted from the PDF. Scanned or image-only "
                "PDFs need OCR, which is not supported."
            )
        raise NoExtractableTextError("The file contains no readable text.")
    return text