import pytest

from app.core.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.services.file_validation import FileType, sanitize_filename, validate_upload

MAX = 10


def test_accepts_txt_and_pdf_case_insensitively() -> None:
    assert validate_upload("notes.txt", b"hello", MAX).file_type is FileType.TXT
    assert validate_upload("REPORT.PDF", b"%PDF-", MAX).file_type is FileType.PDF


@pytest.mark.parametrize(
    "filename", ["a.docx", "noextension", ".pdf", "evil.pdf.exe", "", None]
)
def test_rejects_unsupported_names(filename: str | None) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload(filename, b"data", MAX)


def test_rejects_empty_file() -> None:
    with pytest.raises(EmptyFileError):
        validate_upload("a.txt", b"", MAX)


def test_size_limit_is_inclusive() -> None:
    validate_upload("a.txt", b"x" * MAX, MAX)  # exactly at the limit is fine
    with pytest.raises(FileTooLargeError):
        validate_upload("a.txt", b"x" * (MAX + 1), MAX)


def test_type_is_checked_before_emptiness() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload("a.exe", b"", MAX)


def test_sanitize_strips_directories() -> None:
    assert sanitize_filename("C:\\Users\\me\\doc.pdf") == "doc.pdf"
    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"


def test_sanitize_removes_control_characters() -> None:
    assert sanitize_filename("a\x00b.txt") == "ab.txt"


def test_sanitize_caps_length_but_keeps_extension() -> None:
    name = sanitize_filename("a" * 300 + ".pdf")
    assert len(name) == 255
    assert name.endswith(".pdf")