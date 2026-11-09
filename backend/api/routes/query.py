from fastapi import APIRouter, Depends, HTTPException
from models.request_models import QueryRequest
from models.response_models import QueryResponse
from pipeline.retrieval_workflow import get_graph
from dependencies import get_current_user
from pipeline.prompt import system_prompt
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from fastapi.responses import StreamingResponse
import json
from services.chat_services import save_message
import traceback
from services.pdf_service import get_owned_pdf_or_403, get_pdf_by_share_token
from services.chat_services import get_or_create_session
from services.pdf_service import make_pdf_id_to_str
query_router = APIRouter()


@query_router.post("/pdfs/{pdf_id}/query")
async def query_document(query: QueryRequest, pdf_id: str, current_user: dict = Depends(get_current_user)):
    try:
        pdf_id = make_pdf_id_to_str(pdf_id)
        pdf = await get_owned_pdf_or_403(pdf_id, current_user["id"])
        session = await get_or_create_session(pdf_id, current_user["id"]) 
        thread_id = session["langgraph_thread_id"]
        retrieval_workflow = get_graph() 
        config = {"configurable": {"thread_id": thread_id}}
        # initial_state = {"pdf_id": pdf_id, "query": query.query, "messages": [SystemMessage(content=system_prompt)]}
        initial_state = {"messages": [HumanMessage(query.query)], "pdf_id": pdf_id}
        await save_message(session["id"], "user", query.query)
        async def ai_only_stream():
            full_response = ""
            async for message_chunk, metadata in retrieval_workflow.astream(
                            initial_state,
                            config=config,
                            stream_mode="messages"
                        ):
                            if isinstance(message_chunk, AIMessage):
                                data = {
                                    "delta": message_chunk.content
                                    }
                                full_response += message_chunk.content
                                yield f"data: {json.dumps(data)}\n\n"
            await save_message(session["id"], "assistant", full_response)
            yield 'data: {"delta":"done"}\n\n'
        return StreamingResponse(ai_only_stream(),media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@query_router.post("/shared/{share_token}/query")
async def guest_query_document(query: QueryRequest, share_token: str):
    try:
        if not query.guest_name or not query.guest_name.strip():
            raise HTTPException(status_code=400, detail="guest_name is required")
        pdf = await get_pdf_by_share_token(share_token)
        session = await get_or_create_session(pdf["id"], user_id=None, guest_name=query.guest_name)
        retrieval_workflow = get_graph() 
        config = {"configurable": {"thread_id": session["langgraph_thread_id"]}}
        initial_state = {"pdf_id": pdf["id"], "query": query.query, "messages": [SystemMessage(content=system_prompt)]}
        await save_message(session["id"], "user", query.query)
        async def ai_only_stream():
            full_response = ""
            async for message_chunk, metadata in retrieval_workflow.astream(
                            initial_state,
                            config=config,
                            stream_mode="messages"
                        ):
                            if isinstance(message_chunk, AIMessage):
                                data = {
                                    "delta": message_chunk.content
                                    }
                                full_response += message_chunk.content
                                yield f"data: {json.dumps(data)}\n\n"
            await save_message(session["id"], "assistant", full_response)
            yield 'data: {"delta":"done"}\n\n'
        return StreamingResponse(ai_only_stream(),media_type="text/event-stream")
    except HTTPException:
            raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    