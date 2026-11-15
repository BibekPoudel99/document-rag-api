"""Build tiny valid PDFs for tests, so we don't commit binary fixtures."""

import io

from pypdf import PdfReader, PdfWriter


def make_text_pdf(*page_texts: str) -> bytes:
    """One page per argument; each page holds one line of ASCII text."""
    count = len(page_texts)
    page_numbers = [4 + 2 * i for i in range(count)]  # content stream = page + 1
    kids = " ".join(f"{n} 0 R" for n in page_numbers)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for number, text in zip(page_numbers, page_texts):
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {number + 1} 0 R /Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects.append(
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_position = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode()
    return bytes(out)


def make_blank_pdf() -> bytes:
    """A valid PDF with one page and no text at all (like a scanned document)."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def encrypt_pdf(pdf_bytes: bytes, user_password: str, owner_password: str = "owner") -> bytes:
    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(pdf_bytes)))
    writer.encrypt(user_password=user_password, owner_password=owner_password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()