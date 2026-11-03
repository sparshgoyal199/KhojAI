from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling.chunking import HybridChunker, HierarchicalChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    TripletTableSerializer
)
from sklearn.cluster import KMeans
from docling_core.transforms.serializer.markdown import MarkdownParams
from core.embedding_models import tokenizer
from docling.datamodel.document import ConversionResult
import numpy as np
from fastapi import HTTPException

class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            params=MarkdownParams(
                image_placeholder="<!-- image -->"
            ),
        )

def generate_chunks_payload(chunks: list[dict]) -> list[dict]:
    min_chunk_length = 100
    chunks_payload = []
    total_words = 0
    for idx, chunk in enumerate(chunks):
        text = chunk.text.strip()
        
        if len(text) <= min_chunk_length:
            continue
        
        heading = None
        if hasattr(chunk.meta, "headings") and chunk.meta.headings:
            heading = chunk.meta.headings[0]
            total_words += len(heading.split())
        page_no = None

        if chunk.meta.doc_items:
            first_item = chunk.meta.doc_items[0]

            if first_item.prov:
                page_no = first_item.prov[0].page_no

        filename = None
        if chunk.meta.origin and chunk.meta.origin.filename:
            filename = chunk.meta.origin.filename

        properties = {
            "heading": heading,
            "content": text,
            "page_no": page_no,
            "filename": filename
        }
        chunks_payload.append(properties)
    if len(chunks_payload) == 0:
        raise HTTPException(status_code=400, detail="No valid chunks found after processing. So pdf can not be processed further.")
    avg_doc_length = total_words / len(chunks_payload)
    return (chunks_payload, avg_doc_length)

def create_chunks(structured_doc: ConversionResult) -> list[dict]:
    chunker = HybridChunker(
        tokenizer=tokenizer,
        serializer_provider=MDTableSerializerProvider(),
        merge_peers=True,  # optional, defaults to True
    )
    chunk_iter = chunker.chunk(dl_doc=structured_doc)
    chunks = list(chunk_iter)
    chunks_payload = generate_chunks_payload(chunks)
    return chunks_payload

def k_means_summarised_chunks(total_vectors: int, embedded_chunks: list[list], chunks_payload: list[dict]):
    n_summary_chunks = min(total_vectors,5)
    kmeans = KMeans(n_clusters=n_summary_chunks, random_state=42).fit(embedded_chunks)
    summary_chunks = []
    for cluster_id in range(n_summary_chunks):
        cluster_center = kmeans.cluster_centers_[cluster_id]
        cluster_points_idx = np.where(kmeans.labels_ == cluster_id)[0] # global indices of this cluster's points
        dists = np.linalg.norm(embedded_chunks[cluster_points_idx] - cluster_center, axis=1) # distances of those cluster_id points to centroid
        closest_idx = cluster_points_idx[np.argmin(dists)]
        summary_chunks.append(chunks_payload[closest_idx])
    return summary_chunks