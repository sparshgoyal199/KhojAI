import modal
import os

#MODEL_ID = "BAAI/bge-small-en-v1.5"
#MODEL_ID = "Qwen/Qwen3-Embedding-8B"
MODEL_ID = " BAAI/bge-base-en-v1.5"
image = modal.Image.debian_slim(python_version="3.14").pip_install(
    "torch",
    "sentence-transformers",
    "fastapi[standard]",
    "docling",
    "docling-core",
    "docling-ibm-models",
    "docling-parse"
)

app = modal.App("parsing_and_embedding_generator", image=image)

GPU_CONFIG = "A10G"
CACHE_DIR = "/cache"
cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: cache_vol},
    scaledown_window=60 * 5,
    timeout=60 * 30,
)

@modal.concurrent(max_inputs=15)
class ParsingEmbeddingModel:

    @modal.enter()
    def setup(self):
        import torch
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            MODEL_ID,
            cache_folder=CACHE_DIR,
            #model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda"
        )
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved  = torch.cuda.memory_reserved(0) / 1024**3
        total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"=== VRAM AFTER QWEN Embedding Model ===",'/n')
        print(f"Allocated : {allocated:.2f} GB",'/n')
        print(f"Reserved  : {reserved:.2f} GB",'/n')
        print(f"Total     : {total:.2f} GB",'/n')
        print(f"Free      : {total - reserved:.2f} GB",'/n')

        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["DOCLING_CACHE_DIR"] = CACHE_DIR
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
         
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,   # safe default, CPU threads for pre/post processing
            device=AcceleratorDevice.CUDA
        )
        pipeline_options.do_ocr = False  
        pipeline_options.layout_options.model_spec = DOCLING_LAYOUT_HERON
        pipeline_options.do_table_structure = False
        pipeline_options.table_structure_options.do_cell_matching = False
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False
        pipeline_options.do_formula_enrichment = False
        pipeline_options.do_code_enrichment = False
        pipeline_options.do_picture_description = False
        pipeline_options.do_picture_classification = False
        pipeline_options.generate_parsed_pages = False

        doc_converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        doc_converter.initialize_pipeline(InputFormat.PDF)
        self.docling_obj = doc_converter

        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved  = torch.cuda.memory_reserved(0) / 1024**3
        total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"=== VRAM AFTER DOCLING ===",'/n')
        print(f"Allocated : {allocated:.2f} GB",'/n')
        print(f"Reserved  : {reserved:.2f} GB",'/n')
        print(f"Total     : {total:.2f} GB",'/n')
        print(f"Free      : {total - reserved:.2f} GB",'/n')

    def _encode_query(self, query: str) -> list[float]:
        return self.model.encode(query, normalize_embeddings=True).tolist()

    def _encode_chunks(self, chunks: list[dict]) -> list[list[float]]:
        texts = [chunk.get("content", "") for chunk in chunks]
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def _parse_pdf(self, filename,file_bytes):
        from io import BytesIO
        from docling_core.types.io import DocumentStream
        file_stream = BytesIO(file_bytes)
        docling_input_obj = DocumentStream(name=filename, stream=file_stream)
        structured_doc = self.docling_obj.convert(docling_input_obj)
        return {"docling_document":structured_doc.document.export_to_dict()}
    
    @modal.method()
    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        return self._encode_chunks(chunks)

    @modal.method()
    def embed_query(self, query: str) -> list[float]:
        return self._encode_query(query)
    
    @modal.method()
    def parsing_pdf(self,filename,file_bytes):
        return self._parse_pdf(filename,file_bytes)