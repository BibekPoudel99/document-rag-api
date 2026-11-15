import pytest

from app.core.exceptions import (
    EncryptedPdfError,
    InvalidFileError,
    NoExtractableTextError,
)
from app.services.extraction import extract_text
from app.services.file_validation import FileType
from tests.pdf_factory import encrypt_pdf, make_blank_pdf, make_text_pdf


# ---------- TXT ----------

def test_txt_is_decoded_and_normalised() -> None:
    raw = "héllo\r\n\r\n\r\n\r\nworld  \n".encode("utf-8")
    assert extract_text(raw, FileType.TXT) == "héllo\n\nworld"


def test_txt_with_bom_is_handled() -> None:
    assert extract_text(b"\xef\xbb\xbfhello", FileType.TXT) == "hello"


def test_txt_with_non_utf8_bytes_is_rejected() -> None:
    with pytest.raises(InvalidFileError):
        extract_text("café".encode("cp1252"), FileType.TXT)


def test_txt_with_null_bytes_is_rejected_as_binary() -> None:
    with pytest.raises(InvalidFileError):
        extract_text(b"ab\x00cd", FileType.TXT)


def test_txt_with_only_whitespace_has_no_text() -> None:
    with pytest.raises(NoExtractableTextError):
        extract_text(b"  \n\n  ", FileType.TXT)


# ---------- PDF ----------

def test_pdf_text_is_extracted() -> None:
    assert extract_text(make_text_pdf("Hello PDF world"), FileType.PDF) == "Hello PDF world"


def test_pdf_pages_are_joined_in_order() -> None:
    pdf = make_text_pdf("Page one text", "Page two text")
    assert extract_text(pdf, FileType.PDF) == "Page one text\n\nPage two text"


def test_pdf_without_text_layer_is_reported() -> None:
    with pytest.raises(NoExtractableTextError):
        extract_text(make_blank_pdf(), FileType.PDF)


def test_non_pdf_bytes_with_pdf_extension_are_rejected() -> None:
    with pytest.raises(InvalidFileError):
        extract_text(b"this is not a pdf", FileType.PDF)


def test_pdf_header_with_garbage_body_is_rejected() -> None:
    with pytest.raises(InvalidFileError):
        extract_text(b"%PDF-1.4\nrandom junk\n%%EOF", FileType.PDF)


def test_truncated_pdf_is_rejected() -> None:
    pdf = make_text_pdf("Hello")
    with pytest.raises(InvalidFileError):
        extract_text(pdf[: len(pdf) // 2], FileType.PDF)


def test_password_protected_pdf_is_rejected() -> None:
    locked = encrypt_pdf(make_text_pdf("secret"), user_password="hunter2")
    with pytest.raises(EncryptedPdfError):
        extract_text(locked, FileType.PDF)


def test_pdf_encrypted_with_empty_user_password_is_readable() -> None:
    # Many PDFs are "encrypted" only to restrict printing/copying; they open
    # without a password and should be processed normally.
    restricted = encrypt_pdf(make_text_pdf("open text"), user_password="")
    assert extract_text(restricted, FileType.PDF) == "open text"