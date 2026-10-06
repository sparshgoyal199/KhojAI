from langgraph.graph import StateGraph, START, END
from fastapi import UploadFile,File
from typing import Annotated
from docling_core.types.doc import DoclingDocument
from fastapi import UploadFile
from services.embedding_service import generate_embeddings
from services.document_service import parse_document
from datetime import datetime, timezone
from services.chunk_service import create_chunks, k_means_summarised_chunks
import uuid
from langchain_core.messages import SystemMessage, HumanMessage
from services.retrieval_service import response_generator, creating_user_prompt
from pipeline.prompt import summary_system_prompt
from services.db_service import uploading_file
from services.pdf_service import make_pdf_id_to_str
import numpy as np
from services.vector_service import store_vectors
from typing import TypedDict

class IngestionState(TypedDict):

    pdf_id: str
    user_id: str
    file: Annotated[UploadFile, File(description="A file read as UploadFile")]
    structured_doc: DoclingDocument
    file_bytes: bytes
    chunks_payload: list[dict]
    avg_doc_length: float
    embedded_chunks: list[list]
    summary_chunks: list[dict]
    summary_prompt: str
    summary: str

ingest_graph = StateGraph(IngestionState)

def pdf_id_generation(state: IngestionState):
    pdf_id = uuid.uuid4().hex
    pdf_id = make_pdf_id_to_str(pdf_id)
    return {"pdf_id": pdf_id}

async def document_parsing(state: IngestionState):
    structured_doc, file_bytes = await parse_document(state["file"])
    return {"structured_doc": structured_doc, "file_bytes": file_bytes}

def chunks_creation(state: IngestionState):
    chunks_payload, avg_doc_length = create_chunks(state["structured_doc"])
    return {"chunks_payload": chunks_payload, "avg_doc_length": avg_doc_length}

async def embeddings_generation(state: IngestionState):
    embedded_chunks = await generate_embeddings(
             query=None,
             chunks=state["chunks_payload"],
       )
    return {"embedded_chunks": embedded_chunks}

async def vector_storing(state: IngestionState):
    await store_vectors(
            state["pdf_id"],
            state["chunks_payload"],
            state["embedded_chunks"],
            state["avg_doc_length"],
     )
    return {}

def get_summarised_chunks(state: IngestionState):
    total_vectors = len(state["embedded_chunks"])
    if total_vectors == 0:
        return {"summary_chunks": []}
    
    embedded_chunks = np.array(state["embedded_chunks"])
    summary_chunks = k_means_summarised_chunks(total_vectors=total_vectors, embedded_chunks=embedded_chunks, chunks_payload=state["chunks_payload"])
    return {"summary_chunks": summary_chunks}

def generate_summarize_prompt(state: IngestionState):
    summary_prompt = creating_user_prompt(state["summary_chunks"], query="Summarize the above context", messages=None)
    return {"summary_prompt": summary_prompt}

async def generating_summary_response(state: IngestionState):
    messages = [SystemMessage(content=summary_system_prompt), HumanMessage(content=state["summary_prompt"])]
    response = await response_generator(messages)
    return {"summary": response.content}

async def file_upload(state: IngestionState):
    filename = state["file"].filename
    summary = state["summary"]
    file_bytes = state["file_bytes"]
    user_id = state["user_id"]
    pdf_id = state["pdf_id"]
    pdf_id = pdf_id.replace("-","")
    await uploading_file(pdf_id = pdf_id,  user_id=user_id, filename=filename, file_bytes=file_bytes, summary=summary)
    return {}

ingest_graph.add_node("pdf_id_generation", pdf_id_generation)
ingest_graph.add_node("document_parsing", document_parsing)
ingest_graph.add_node("chunks_creation", chunks_creation)
ingest_graph.add_node("embeddings_generation", embeddings_generation)
ingest_graph.add_node("vector_storing", vector_storing)
ingest_graph.add_node("get_summarised_chunks", get_summarised_chunks)
ingest_graph.add_node("generate_summarize_prompt", generate_summarize_prompt)
ingest_graph.add_node("generating_summary_response", generating_summary_response)
ingest_graph.add_node("file_upload", file_upload)

# ingest_graph.add_edge(START, "pdf_id_generation")
# ingest_graph.add_edge("pdf_id_generation", "document_parsing")
# ingest_graph.add_edge("document_parsing", "chunks_creation")
# ingest_graph.add_edge("chunks_creation", "embeddings_generation")
# ingest_graph.add_edge("embeddings_generation", "vector_storing")
# ingest_graph.add_edge("vector_storing", "get_summarised_chunks")
# ingest_graph.add_edge("get_summarised_chunks", "generate_summarize_prompt")
# ingest_graph.add_edge("generate_summarize_prompt", "generating_summary_response")
# ingest_graph.add_edge("generating_summary_response", "file_upload")
# ingest_graph.add_edge("file_upload", END)

ingest_graph.add_edge(START, "pdf_id_generation")
ingest_graph.add_edge("pdf_id_generation", "document_parsing")
ingest_graph.add_edge("document_parsing", "chunks_creation")
ingest_graph.add_edge("chunks_creation", "embeddings_generation")
ingest_graph.add_edge("embeddings_generation", "vector_storing")
ingest_graph.add_edge("embeddings_generation", "get_summarised_chunks")
ingest_graph.add_edge("get_summarised_chunks", "generate_summarize_prompt")
ingest_graph.add_edge("generate_summarize_prompt", "generating_summary_response")
ingest_graph.add_edge(["vector_storing", "generating_summary_response"], "file_upload")
ingest_graph.add_edge("file_upload", END)


ingestion_workflow = ingest_graph.compile()