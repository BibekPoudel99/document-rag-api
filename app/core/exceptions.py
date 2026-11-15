"""Domain errors. Each carries the HTTP status and a stable machine-readable code.

This module knows nothing about FastAPI; app/api/error_handlers.py maps these
errors to HTTP responses.
"""


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class EmptyFileError(AppError):
    status_code = 422
    code = "empty_file"


class InvalidFileError(AppError):
    """Corrupt PDF, wrong encoding, or content that doesn't match the extension."""

    status_code = 422
    code = "invalid_file"


class EncryptedPdfError(AppError):
    status_code = 422
    code = "encrypted_pdf"


class NoExtractableTextError(AppError):
    status_code = 422
    code = "no_extractable_text"