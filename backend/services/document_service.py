from fastapi import UploadFile, HTTPException
# from io import BytesIO
# from docling_core.types.io import DocumentStream
# from core.docling import doc_converter
import logging
# from docling_core.types.doc import DoclingDocument
from collections import Counter
from docling_core.types.doc import DocItemLabel, SectionHeaderItem
from docling.datamodel.document import ConversionResult
import pickle
from docling_core.types.doc import DoclingDocument
import modal

#EmbeddingModel = modal.Cls.from_name("embeddings-generator", "EmbeddingModel")
parsing_and_embedding_model = modal.Cls.from_name("parsing_and_embedding_generator", "ParsingEmbeddingModel")
log = logging.getLogger(__name__)
def validate_document(doc: ConversionResult):
    """
    Collect all SECTION_HEADER texts that appear multiple times
    — these are likely running page headers.
    """
    header_texts = []
    threshold = 0.08
    min_repeat = 4
    for item, _ in doc.iterate_items():
        if item.label == DocItemLabel.SECTION_HEADER:
            header_texts.append(item.text.strip())

    counts = Counter(header_texts)
    repeating_headers = {text for text, count in counts.items() if count >= min_repeat}
    validate_doc = remove_false_headers_from_doc(doc, repeating_headers, threshold)
    return validate_doc

def is_false_page_header(item, doc, repeating_headers: set, threshold: float = 0.08) -> bool:
    """Check if an item is a false page header (repeated text or at top of page)."""
    if item.text.strip() in repeating_headers:
        return True
    for prov in item.prov:
        page = doc.pages.get(prov.page_no)
        if page is None:
            continue
        if prov.bbox.t >= page.size.height * (1 - threshold):
            return True
    return False

def remove_false_headers_from_doc(doc, repeating_headers: set, threshold: float = 0.08):
    """
    Removes false page headers from the docling document.
    Re-parents their children to the preceding real heading before deleting.
    """

    # ── Step 1: Identify all false headers ──────────────────────────────────
    false_headers = []
    false_header_refs = set()

    for item, _ in doc.iterate_items():
        if item.label == DocItemLabel.SECTION_HEADER:
            if is_false_page_header(item, doc, repeating_headers, threshold):
                false_headers.append(item)
                false_header_refs.add(item.self_ref)
                log.info(f"Flagged for removal → '{item.text.strip()}'")

    if not false_headers:
        log.info("No false headers found to remove.")
        return doc

    # ── Step 2: Re-parent children of each false header ─────────────────────
    for header in false_headers:
        if not header.children:
            continue # no children to re-parent, skip

        # Find the parent node of this false header
        parent_node = header.parent.resolve(doc=doc)

        # Find the index of this false header among its parent's children
        header_ref = header.get_ref()
        header_index = None
        for i, child_ref in enumerate(parent_node.children):
            if child_ref.cref == header_ref.cref:
                header_index = i
                break

        if header_index is None:
            log.warning(f"Could not find header '{header.text.strip()}' in parent's children, skipping re-parent.")
            continue

        # Walk backwards through siblings to find the nearest real heading
        real_heading = None
        for i in range(header_index - 1, -1, -1):
            sibling = parent_node.children[i].resolve(doc)
            if (isinstance(sibling, SectionHeaderItem)
                    and sibling.self_ref not in false_header_refs):
                real_heading = sibling
                break

        if real_heading:
            # Move each child of the false header to the real heading
            for child_ref in list(header.children): # copy list since _move_subtree modifies it
                child = child_ref.resolve(doc)
                doc._move_subtree(old_subroot=child, new_subroot=real_heading)
            log.info(f"Re-parented children of '{header.text.strip()}' → under '{real_heading.text.strip()}'")
        else:
            # No preceding real heading found — shift children up to the parent level
            log.info(f"No preceding heading found for '{header.text.strip()}', shifting children up.")
            doc._shift_up(old_subroot=header)
            # _shift_up already removes the header from its parent's children,
            # so we should NOT delete it again via delete_items.
            # Remove it from our false_headers list to avoid double-removal.
            false_header_refs.discard(header.self_ref)

    # ── Step 3: Delete the now-childless false headers ──────────────────────
    # Only delete headers that weren't already removed by _shift_up
    headers_to_delete = [h for h in false_headers if h.self_ref in false_header_refs]
    if headers_to_delete:
        doc.delete_items(node_items=headers_to_delete)

    log.info(f"Successfully removed {len(false_headers)} false header(s) from document.")
    return doc

async def parse_document(file: UploadFile):
    filename = file.filename
    file_bytes = file.file.read()
    file_extension = file.content_type.split("/")[-1].lower()
    # if file_extension == "octet-stream":
    #     valid_structured_doc = pickle.loads(file_bytes)
    #     return valid_structured_doc
    if file_extension not in ["pdf"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
    modal_obj = parsing_and_embedding_model()
    doc_obj = await modal_obj.parsing_pdf.remote.aio(filename,file_bytes)
    docling_doc = doc_obj["docling_document"]
    structured_doc = DoclingDocument.model_validate(docling_doc)
    valid_structured_doc = validate_document(structured_doc)
    return (valid_structured_doc, file_bytes)