import html
import io
import logging
import os
import re
from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser

try:
    import pymupdf
except ImportError:  # pragma: no cover
    pymupdf = None

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None
    ImageOps = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

if pytesseract is not None:
    configured_tesseract = os.environ.get('TESSERACT_CMD')
    if configured_tesseract:
        pytesseract.pytesseract.tesseract_cmd = configured_tesseract
    elif os.name == 'nt':
        for tesseract_path in (
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ):
            if os.path.isfile(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                break


MAX_UPLOAD_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.txt', '.eml', '.pdf', '.png', '.jpg', '.jpeg', '.webp'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
IMAGE_FORMATS = {
    '.png': 'PNG',
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.webp': 'WEBP',
}
logger = logging.getLogger(__name__)


def safe_file_name(filename):
    name = (filename or '').strip()
    if not name:
        return 'uploaded_file'
    return os.path.basename(name).replace('\\', '/').split('/')[-1]


def _decode_header_value(value):
    if not value:
        return ''
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return str(value)


def _strip_html(raw_html):
    if not raw_html:
        return ''
    cleaned = re.sub(r'<style.*?>.*?</style>', ' ', raw_html, flags=re.I | re.S)
    cleaned = re.sub(r'<script.*?>.*?</script>', ' ', cleaned, flags=re.I | re.S)
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


def _find_urls(text):
    if not text:
        return []
    matches = re.findall(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', text, flags=re.IGNORECASE)
    cleaned = []
    for match in matches:
        match = match.rstrip(').,;:!?')
        if match not in cleaned:
            cleaned.append(match)
    return cleaned


def _decode_text(content):
    for encoding in ('utf-8-sig', 'utf-16', 'cp1252', 'latin-1'):
        try:
            text = content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        control_characters = sum(
            not character.isprintable() and character not in '\r\n\t\f'
            for character in text
        )
        if text and control_characters / len(text) > 0.05:
            continue
        return text
    raise ValueError('The text file is not readable text or uses an unsupported encoding.')


def validate_file_upload(uploaded_file):
    if uploaded_file is None:
        raise ValueError('Please choose a file to scan.')

    file_name = safe_file_name(getattr(uploaded_file, 'name', ''))
    if not file_name:
        raise ValueError('The uploaded file is missing a valid name.')

    extension = os.path.splitext(file_name)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            'Unsupported file type. Please upload a .txt, .eml, .pdf, .png, .jpg, .jpeg, or .webp file.'
        )

    uploaded_file.seek(0, os.SEEK_END)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)
    if file_size <= 0:
        raise ValueError('The uploaded file is empty.')
    if file_size > MAX_UPLOAD_SIZE:
        raise ValueError('The uploaded file exceeds the 10 MB size limit.')

    content = uploaded_file.read()
    uploaded_file.seek(0)

    if extension == '.txt':
        _decode_text(content)
    elif extension == '.eml':
        try:
            message = BytesParser(policy=policy.default).parsebytes(content)
            if not any(message.get(header) for header in ('From', 'To', 'Subject', 'Date', 'Message-ID')):
                raise ValueError('The file does not contain recognizable email headers.')
        except Exception as exc:
            raise ValueError('The email file is malformed or corrupted.') from exc
    elif extension == '.pdf':
        if pymupdf is None:
            raise ValueError('PDF scanning requires PyMuPDF. Install the project dependencies and try again.')
        try:
            document = pymupdf.open(stream=content, filetype='pdf')
            if document.is_encrypted:
                document.close()
                raise ValueError('This PDF is password-protected and cannot be processed securely.')
            document.close()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError('The PDF file appears corrupted or is not a valid PDF.') from exc
    elif extension in IMAGE_EXTENSIONS:
        if Image is None:
            raise ValueError('Image scanning requires Pillow. Install the project dependencies and try again.')
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.format != IMAGE_FORMATS[extension]:
                    raise ValueError('The image content does not match its file extension.')
                image.verify()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError('The image file is invalid or corrupted.') from exc

    return file_name, extension


def _extract_email_attachment_text(file_name, content_bytes):
    lower_name = (file_name or '').lower()
    if lower_name.endswith('.eml'):
        info = extract_email_text(content_bytes)
        return info.get('text', ''), info.get('urls', []), info.get('warnings', [])
    if lower_name.endswith(('.txt', '.pdf', '.png', '.jpg', '.jpeg', '.webp')):
        info = extract_file_text(file_name, content_bytes)
        return info.get('text', ''), info.get('urls', []), info.get('warnings', [])
    return '', [], []


def extract_email_text(raw_email_bytes):
    message = BytesParser(policy=policy.default).parsebytes(raw_email_bytes)
    sender = _decode_header_value(message.get('From', ''))
    subject = _decode_header_value(message.get('Subject', ''))
    warnings = []
    body_parts = []
    attachment_text = []
    urls = []

    for part in message.walk():
        if part.is_multipart():
            continue

        file_name = part.get_filename()
        if file_name:
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            extracted_text, attachment_urls, attachment_warnings = _extract_email_attachment_text(file_name, payload)
            if extracted_text:
                attachment_text.append(extracted_text)
            if attachment_urls:
                urls.extend(attachment_urls)
            if attachment_warnings:
                warnings.extend(attachment_warnings)
            continue

        content_type = part.get_content_type()
        payload = part.get_payload(decode=True)
        if payload is None:
            continue

        try:
            charset = part.get_content_charset() or 'utf-8'
            decoded = payload.decode(charset, errors='replace')
        except LookupError:
            decoded = payload.decode('utf-8', errors='replace')

        if content_type == 'text/html':
            decoded = _strip_html(decoded)
        if content_type in {'text/plain', 'text/html'}:
            if decoded.strip():
                body_parts.append(decoded)

    combined_text = '\n'.join(part for part in body_parts if part.strip())
    printable_text = '\n'.join(filter(None, [subject, sender, combined_text, *attachment_text]))
    urls.extend(_find_urls(printable_text))
    urls = list(dict.fromkeys(urls))

    return {
        'sender': sender,
        'subject': subject,
        'body': combined_text,
        'text': printable_text,
        'urls': urls,
        'warnings': warnings,
    }


def _ocr_image(image):
    if pytesseract is None:
        return '', [
            'OCR requires pytesseract and the native Tesseract engine. Install the project dependencies and Tesseract, then retry.'
        ]
    if Image is None or ImageOps is None:
        return '', ['OCR requires Pillow. Install the project dependencies and retry.']

    try:
        pytesseract.get_tesseract_version()
        grayscale = ImageOps.grayscale(image)
        enhanced = ImageOps.autocontrast(grayscale)
        enhanced = enhanced.resize((enhanced.width * 2, enhanced.height * 2))
        text = pytesseract.image_to_string(enhanced, config='--psm 6')
        return text.strip(), []
    except Exception as exc:  # pragma: no cover
        logger.warning('OCR could not run: %s', exc)
        return '', [
            'OCR is unavailable. Install the Tesseract OCR engine and pytesseract, then configure the Tesseract executable.'
        ]


def extract_image_text(image_bytes, file_name='image.png'):
    warnings = []
    if Image is None or ImageOps is None:
        return {'text': '', 'warnings': ['Image processing support is not available.'], 'urls': []}

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert('RGB')
            text, ocr_warnings = _ocr_image(image)
            warnings.extend(ocr_warnings)
            cleaned_text = re.sub(r'\s+', ' ', text).strip()
            if not cleaned_text:
                warnings.append('Insufficient readable text to make a reliable prediction.')
            return {
                'text': cleaned_text,
                'warnings': warnings,
                'urls': _find_urls(cleaned_text),
                'ocr_used': pytesseract is not None and not ocr_warnings,
            }
    except Exception as exc:
        logger.warning('Uploaded image could not be processed: %s', exc)
        return {'text': '', 'warnings': ['The image could not be processed.'], 'urls': [], 'ocr_used': False}


def extract_pdf_text(pdf_bytes):
    warnings = []
    if pymupdf is None:
        return {
            'text': '',
            'warnings': ['PDF scanning requires PyMuPDF. Install the project dependencies and try again.'],
            'urls': [],
            'ocr_used': False,
        }

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype='pdf')
    except Exception as exc:
        logger.warning('Uploaded PDF could not be opened: %s', exc)
        return {'text': '', 'warnings': ['The PDF could not be opened. It may be corrupted.'], 'urls': [], 'ocr_used': False}

    if document.is_encrypted:
        document.close()
        return {
            'text': '',
            'warnings': ['This PDF is password-protected and cannot be processed securely.'],
            'urls': [],
            'ocr_used': False,
        }

    text_chunks = []
    total_pages = min(len(document), 30)
    if len(document) > total_pages:
        warnings.append('The PDF has more than 30 pages; only the first 30 pages were scanned.')

    ocr_used = False
    try:
        for page_index in range(total_pages):
            page = document[page_index]
            page_text = page.get_text('text')
            if page_text and page_text.strip():
                text_chunks.append(page_text)

            if len((page_text or '').strip()) >= 40 and not page.get_images():
                continue

            if pytesseract is None or Image is None:
                warnings.append(
                    f'Page {page_index + 1} may contain image-based text. Install Tesseract OCR and pytesseract to scan it.'
                )
                continue

            try:
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
                ocr_text, ocr_warnings = _ocr_image(image)
                if ocr_warnings:
                    warnings.extend(ocr_warnings)
                else:
                    ocr_used = True
                if ocr_text.strip():
                    text_chunks.append(ocr_text)
            except Exception as exc:  # pragma: no cover
                logger.warning('Could not OCR PDF page %s: %s', page_index + 1, exc)
                warnings.append(f'Could not OCR page {page_index + 1}.')
    finally:
        document.close()

    final_text = ' '.join(text_chunks)
    cleaned_text = re.sub(r'\s+', ' ', final_text).strip()
    if not cleaned_text:
        warnings.append('No readable text could be extracted from this PDF.')

    urls = _find_urls(cleaned_text)
    return {'text': cleaned_text, 'warnings': warnings, 'urls': urls, 'ocr_used': ocr_used}


def extract_file_text(file_name, file_bytes):
    name = safe_file_name(file_name)
    extension = os.path.splitext(name)[1].lower()

    if extension == '.txt':
        text = _decode_text(file_bytes).strip()
        return {'text': text, 'warnings': [], 'urls': _find_urls(text), 'ocr_used': False}
    if extension == '.eml':
        return extract_email_text(file_bytes)
    if extension == '.pdf':
        return extract_pdf_text(file_bytes)
    if extension in IMAGE_EXTENSIONS:
        return extract_image_text(file_bytes, name)
    return {'text': '', 'warnings': ['This file type is not supported for scanning.'], 'urls': []}


def extract_text_from_uploaded_file(uploaded_file, file_name=None):
    file_name = safe_file_name(file_name or getattr(uploaded_file, 'name', ''))
    extension = os.path.splitext(file_name)[1].lower()
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    if extension == '.txt':
        text = _decode_text(file_bytes).strip()
        return {'text': text, 'warnings': [], 'urls': _find_urls(text), 'ocr_used': False}
    if extension == '.eml':
        return extract_email_text(file_bytes)
    if extension == '.pdf':
        return extract_pdf_text(file_bytes)
    if extension in IMAGE_EXTENSIONS:
        return extract_image_text(file_bytes, file_name)
    raise ValueError(
        'Unsupported file type. Please upload a .txt, .eml, .pdf, .png, .jpg, .jpeg, or .webp file.'
    )
