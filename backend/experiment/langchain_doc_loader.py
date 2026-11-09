from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("pdfs/JudgeFlow.pdf")
documents = loader.load()

# with open("archive/extracted_output.txt", "w", encoding="utf-8") as file:
#     for doc in documents:
#         file.write(f"Metadata: {doc.metadata}\n")
#         file.write(f"Content:\n{doc.page_content}\n")
#         file.write("\n" + "=" * 100 + "\n\n")

with open("extracted_output.md", "w", encoding="utf-8") as file:
    file.write("# Extracted PDF Content\n\n")

    for doc in documents:
        file.write(f"## Page {doc.metadata['page'] + 1}\n\n")
        file.write(f"**Source:** {doc.metadata['source']}\n\n")
        file.write(doc.page_content)
        file.write("\n\n---\n\n")

print("PDF content saved to Markdown file!")

print("PDF content saved successfully!")