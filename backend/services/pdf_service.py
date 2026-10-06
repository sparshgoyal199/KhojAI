from fastapi import HTTPException
import core.db as core_db
import uuid
from typing import Optional

async def get_owned_pdf_or_403(pdf_id: str, user_id: str) -> dict:
    result = await core_db.supabase_client.table("pdfs").select("*").eq("id", pdf_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="PDF not found")
    pdf = result.data[0]
    if pdf["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return pdf


async def get_pdf_by_share_token(share_token: str) -> dict:
    result = await core_db.supabase_client.table("pdfs").select("*").eq("share_token", share_token).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Invalid or expired share link")
    return result.data[0]


async def generate_share_link(pdf_id: str, user_id: str, base_url: str) -> dict:
    pdf = await get_owned_pdf_or_403(pdf_id, user_id)

    if pdf.get("share_token"):
        token = pdf["share_token"]  
    else:
        token = str(uuid.uuid4())
        await core_db.supabase_client.table("pdfs").update({"share_token": token}).eq("id", pdf_id).execute()

    return {
        "share_token": token,
        "share_url": f"{base_url}/shared/{token}",
    }

async def get_pdfs_for_user(user_id: str) -> list[dict]:
    result = (
        await core_db.supabase_client.table("pdfs")
        .select("id, filename, upload_date, summary")
        .eq("user_id", user_id)
        .order("upload_date", desc=True)
        .execute()
    )
    return result.data


async def _get_file_signed_url(storage_path: str) -> Optional[str]:
    if not storage_path:
        return None
    result = await core_db.supabase_client.storage.from_("pdfs").create_signed_url(storage_path, 3600) #1 hour
    return result.get("signedURL") or result.get("signed_url")


async def get_pdf_detail_for_owner(pdf_id: str, user_id: str) -> dict:
    pdf = await get_owned_pdf_or_403(pdf_id, user_id)
    pdf["file_url"] = await _get_file_signed_url(pdf.get("storage_path"))
    return pdf


async def get_pdf_detail_for_guest(share_token: str) -> dict:
    pdf = await get_pdf_by_share_token(share_token)   # already exists
    pdf["file_url"] = await _get_file_signed_url(pdf.get("storage_path"))
    return pdf

async def get_searched_pdf(q: str, user_id: str):
    try:
        response = (
            await core_db.supabase_client
            .rpc("search_uploads", {"search_term": q, "uid": user_id})
            .execute()
        )
        return response.data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def make_pdf_id_to_str(pdf_id):
    pdf_id = pdf_id.replace("-","")
    return pdf_id