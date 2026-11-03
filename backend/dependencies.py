from fastapi import HTTPException, Header
import core.db as core_db
from core.security import decode_access_token

async def get_current_user(authorization: str = Header(...)) -> dict:
    try:
        token = authorization.replace("Bearer ", "")
        payload = decode_access_token(token)
        user_id = payload.get("sub")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    result = await core_db.supabase_client.table("users").select("*").eq("id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=401, detail="User not found")

    return result.data[0]  