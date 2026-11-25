import logfire

def parse_office(file_path: str):
    """
    Parses Office documents (.docx, .pptx) without loading Unstructured.
    This avoids native libraries that Windows application control may block.
    """
    with logfire.span("Office Document Parsing", filename=file_path):
        try:
            lower_path = file_path.lower()

            if lower_path.endswith(".docx"):
                from docx import Document

                document = Document(file_path)
                text_parts = []

                for paragraph in document.paragraphs:
                    if paragraph.text.strip():
                        text_parts.append(paragraph.text)

                for table in document.tables:
                    for row in table.rows:
                        row_text = " | ".join(
                            cell.text.strip()
                            for cell in row.cells
                            if cell.text.strip()
                        )
                        if row_text:
                            text_parts.append(row_text)

            elif lower_path.endswith(".pptx"):
                from pptx import Presentation

                presentation = Presentation(file_path)
                text_parts = []

                for slide_number, slide in enumerate(presentation.slides, start=1):
                    slide_parts = []
                    for shape in slide.shapes:
                        if getattr(shape, "has_text_frame", False):
                            text = shape.text.strip()
                            if text:
                                slide_parts.append(text)
                        if getattr(shape, "has_table", False):
                            for row in shape.table.rows:
                                row_text = " | ".join(
                                    cell.text.strip()
                                    for cell in row.cells
                                    if cell.text.strip()
                                )
                                if row_text:
                                    slide_parts.append(row_text)

                    if slide_parts:
                        text_parts.append(
                            f"Slide {slide_number}\n" + "\n".join(slide_parts)
                        )

            else:
                raise ValueError(f"Unsupported Office file: {file_path}")

            full_text = "\n".join(text_parts)
            
            if not full_text.strip():
                logfire.warning(f"No text extracted from {file_path}")
            else:
                logfire.info(f"Successfully parsed {len(full_text)} characters")

            return full_text
        except Exception as e:
            logfire.error(f"Office Parse Failed: {e}")
            raise e