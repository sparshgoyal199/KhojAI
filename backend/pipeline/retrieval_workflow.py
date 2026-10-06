from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from typing import Annotated
from services.retrieval_service import response_generator, retrieve_relevant_chunks, creating_user_prompt
from services.embedding_service import generate_embeddings
from langchain_core.messages import SystemMessage
from pipeline.prompt import system_prompt
from core.checkpointer import checkpointer 

class retrievalState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]
    pdf_id: str
    query: str
    embedded_query: list
    relevant_chunks_payload: list[dict]
    user_prompt: str
    #response: str

retrieve_garph = StateGraph(retrievalState)

async def embeddings_generation(state: retrievalState):
    embedded_query = await generate_embeddings(query=state["query"],chunks=None)
    return {"embedded_query": embedded_query}

async def retrieving_chunks(state: retrievalState):
    relevant_chunks_payload = await retrieve_relevant_chunks(state["pdf_id"], state["embedded_query"], state["query"])
    return {"relevant_chunks_payload": relevant_chunks_payload}

def prompt_formatting(state: retrievalState):
    user_prompt = creating_user_prompt(state["relevant_chunks_payload"], state["query"], state["messages"])
    return {"user_prompt": user_prompt, "messages": HumanMessage(content=state["query"])}

async def generating_response(state: retrievalState):
    response = await response_generator([SystemMessage(content=system_prompt), HumanMessage(content=state["user_prompt"])])
    return {"messages": [response]}

retrieve_garph.add_node("embeddings_generation", embeddings_generation)
retrieve_garph.add_node("retrieving_chunks", retrieving_chunks)
retrieve_garph.add_node("prompt_formatting", prompt_formatting)
retrieve_garph.add_node("generating_response", generating_response)

retrieve_garph.add_edge(START, "embeddings_generation")
retrieve_garph.add_edge("embeddings_generation", "retrieving_chunks")
retrieve_garph.add_edge("retrieving_chunks", "prompt_formatting")
retrieve_garph.add_edge("prompt_formatting", "generating_response")
retrieve_garph.add_edge("generating_response", END)

_graph = None  # lazy, module-level placeholder

def get_graph():
    global _graph
    if _graph is None:
        from core.checkpointer import checkpointer
        _graph = retrieve_garph.compile(checkpointer=checkpointer) 
    return _graph
#retrieval_workflow = retrieve_garph.compile(checkpointer=checkpointer)