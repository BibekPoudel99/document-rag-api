from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from app.core.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)

_MAX_FILENAME_LENGTH = 255


class FileType(StrEnum):
    PDF = "pdf"
    TXT = "txt"


_EXTENSIONS: dict[str, FileType] = {".pdf": FileType.PDF, ".txt": FileType.TXT}


@dataclass(frozen=True)
class ValidatedUpload:
    filename: str
    file_type: FileType


def sanitize_filename(filename: str | None) -> str:
    """Return a safe display name: no directories, no control characters."""
    # Some clients send full Windows paths; normalise separators, keep the last part.
    name = PurePosixPath((filename or "").replace("\\", "/")).name
    name = "".join(char for char in name if char.isprintable())
    return name[-_MAX_FILENAME_LENGTH:]  # keep the end so the extension survives


def validate_upload(
    filename: str | None, content: bytes, max_bytes: int
) -> ValidatedUpload:
    """Check type, size and emptiness. Does not look inside the file."""
    safe_name = sanitize_filename(filename)
    file_type = _EXTENSIONS.get(PurePosixPath(safe_name).suffix.lower())
    if file_type is None:
        raise UnsupportedFileTypeError("Only .pdf and .txt files are supported.")
    if len(content) == 0:
        raise EmptyFileError("The uploaded file is empty.")
    if len(content) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise FileTooLargeError(f"The file exceeds the maximum size of {limit_mb:g} MB.")
    return ValidatedUpload(filename=safe_name, file_type=file_type)