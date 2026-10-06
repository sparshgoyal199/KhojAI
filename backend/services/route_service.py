# from langchain_core.messages import SystemMessage
# from core.llm import llm
# from models.router import RouterResponse
# from pipeline.prompt import ROUTER_SYSTEM_PROMPT

# async def route_query_service(messages: list, query: str):
#     if len(messages) == 0: 
#         return False
    
#     history_msg = []
#     for msg in messages:
#         if isinstance(msg, SystemMessage):
#             continue
#         history_msg.append(msg)
    
#     history_text = "\n".join(
#         f"{msg.type}: {msg.content}" for msg in history_msg
#     )

#     user_msg = HumanMessage(content=f"""
#         ## Conversation History:
#         {history_text}

#         ## User's Question:
#         {query}
#         """)
    
#     router_llm = llm.with_structured_output(RouterResponse)
#     response = await router_llm.ainvoke([
#         SystemMessage(content=ROUTER_SYSTEM_PROMPT),
#         user_msg
#     ])
#     return response
