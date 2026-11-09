from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from typing import Annotated
from services.retrieval_service import response_generator, retrieve_relevant_chunks, creating_user_prompt
from services.embedding_service import generate_embeddings
from langchain_core.messages import SystemMessage
from pipeline.prompt import system_prompt
from langchain.tools import tool, ToolRuntime
from langgraph.prebuilt import ToolNode, tools_condition
from core.checkpointer import checkpointer 

# class retrievalState(TypedDict):

#     messages: Annotated[list[BaseMessage], add_messages]
#     pdf_id: str
#     query: str
#     embedded_query: list
#     relevant_chunks_payload: list[dict]
#     user_prompt: str
#     #response: str


class retrievalState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]
    pdf_id: str
    
retrieve_garph = StateGraph(retrievalState)


@tool
async def rag_tool(query: str, runtime: ToolRuntime):
    """Search the knowledge base for passages relevant to a query.

    USE WHEN: the user asks a factual, definitional, explanatory, or
    document-specific question that cannot be answered from the conversation
    alone — e.g. "what does it say about X", "explain Y", "find the section
    on Z", "summarize the part about W".

    DO NOT USE: for greetings, thanks, chit-chat, or questions about your own
    capabilities; and not when the needed information is already present in
    earlier tool results or the conversation.

    QUERY FORMAT: this search is stateless — it does not see the conversation.
    Write `query` as one self-contained search phrase that restates the full
    intent. Resolve all pronouns and references ("it", "that method", "the
    second one") into explicit terms, and drop conversational filler. Prefer
    a few focused noun phrases over a full question sentence.
    Example — instead of "what about its limitations?", use
    "limitations of the transformer attention mechanism".

    RETURNS: a list of relevant passages. Each item has `text` (the passage
    content) and `page` (the page number it came from). Use `page` for inline
    [p. X] citations. Returns an empty list when nothing relevant is found —
    in that case do not invent an answer; either retry once with a
    reformulated query or tell the user the knowledge base has no matching
    content.
    """
    print("Here is the query refined by LLM: ")
    print(query)
    print("*" * 65)
    pdf_id = runtime.state["pdf_id"]
    embedded_query = await generate_embeddings(query=query,chunks=None)
    relevant_chunks_payload = await retrieve_relevant_chunks(pdf_id, embedded_query, query)
    print("Here is the relevant chunks: ")
    print(relevant_chunks_payload)
    print("*" * 65)
    return relevant_chunks_payload
    #user_prompt = creating_user_prompt(state["relevant_chunks_payload"], state["query"], state["messages"])

tools = [rag_tool]
tool_node = ToolNode(tools)
# async def embeddings_generation(state: retrievalState):
#     embedded_query = await generate_embeddings(query=state["query"],chunks=None)
#     return {"embedded_query": embedded_query}

# async def retrieving_chunks(state: retrievalState):
#     relevant_chunks_payload = await retrieve_relevant_chunks(state["pdf_id"], state["embedded_query"], state["query"])
#     return {"relevant_chunks_payload": relevant_chunks_payload}

# def prompt_formatting(state: retrievalState):
#     user_prompt = creating_user_prompt(state["relevant_chunks_payload"], state["query"], state["messages"])
#     return {"user_prompt": user_prompt, "messages": HumanMessage(content=state["query"])}

# async def generating_response(state: retrievalState):
#     response = await response_generator([SystemMessage(content=system_prompt), HumanMessage(content=state["user_prompt"])])
#     return {"messages": [response]}

async def chat_model(state: retrievalState):
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = await response_generator(messages=messages)
    return {"messages": [response]}

# retrieve_garph.add_node("embeddings_generation", embeddings_generation)
# retrieve_garph.add_node("retrieving_chunks", retrieving_chunks)
# retrieve_garph.add_node("prompt_formatting", prompt_formatting)
# retrieve_garph.add_node("generating_response", generating_response)

# retrieve_garph.add_edge(START, "embeddings_generation")
# retrieve_garph.add_edge("embeddings_generation", "retrieving_chunks")
# retrieve_garph.add_edge("retrieving_chunks", "prompt_formatting")
# retrieve_garph.add_edge("prompt_formatting", "generating_response")
# retrieve_garph.add_edge("generating_response", END)

retrieve_garph.add_node("chat_model", chat_model)
retrieve_garph.add_node("tools", tool_node)

retrieve_garph.add_edge(START, "chat_model")
retrieve_garph.add_conditional_edges("chat_model", tools_condition)
retrieve_garph.add_edge('tools', 'chat_model')

_graph = None  # lazy, module-level placeholder

def get_graph():
    global _graph
    if _graph is None:
        from core.checkpointer import checkpointer
        _graph = retrieve_garph.compile(checkpointer=checkpointer) 

    return _graph
#retrieval_workflow = retrieve_garph.compile(checkpointer=checkpointer)