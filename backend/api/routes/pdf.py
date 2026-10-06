from fastapi import APIRouter, Depends, Request,Query
from services.pdf_service import generate_share_link
from models.pdf import ShareResponse
from dependencies import get_current_user
from models.pdf import PdfListItem, PdfDetail, SharedPdfDetail
from services.pdf_service import (
    get_pdfs_for_user,
    get_pdf_detail_for_owner,
    get_pdf_detail_for_guest,
    get_searched_pdf
)
from services.pdf_service import make_pdf_id_to_str

pdf_router = APIRouter(prefix="/pdfs", tags=["pdfs"])

@pdf_router.post("/{pdf_id}/share", response_model=ShareResponse)
async def share_pdf(pdf_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    pdf_id = make_pdf_id_to_str(pdf_id)
    base_url = str(request.base_url).rstrip("/")
    return await generate_share_link(pdf_id, current_user["id"], base_url)

@pdf_router.get("", response_model=list[PdfListItem])
async def list_my_pdfs(current_user: dict = Depends(get_current_user)):
    return await get_pdfs_for_user(current_user["id"])

@pdf_router.get("/shared/{share_token}", response_model=SharedPdfDetail)
async def view_shared_pdf(share_token: str):
    return await get_pdf_detail_for_guest(share_token)

@pdf_router.get("/search", response_model=list[PdfListItem])
async def search_pdfs(
    q: str = Query(..., min_length=1),
    current_user: dict = Depends(get_current_user),
):
    return await get_searched_pdf(q, current_user["id"])

@pdf_router.get("/{pdf_id}", response_model=PdfDetail)
async def view_my_pdf(pdf_id: str, current_user: dict = Depends(get_current_user)):
    pdf_id = make_pdf_id_to_str(pdf_id)
    return await get_pdf_detail_for_owner(pdf_id, current_user["id"])
