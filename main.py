# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional, Dict
# from partnerable_agent import PartnerableAgent
# import uvicorn
#
# # agent = PartnerableAgent(
# #     llm=None,  # Replace with OpenAI LLM client if applicable
# #     tools=tools,
# #     verbose=True,
# #     agent_name="Socrates",
# #     agent_role="Reflective Philosopher",
# # )
#
# # Initialize FastAPI app
# app = FastAPI()
#
# # Enable CORS for all origins, all methods, all headers
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Or specify specific domains like ["http://example.com"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# # Mock Tools
# def mock_tool(input_text: str) -> str:
#     return f"Processed input: {input_text}"
#
#
# tools = {"MockTool": mock_tool}
#
# # Initialize Socrates agent
# agent = PartnerableAgent(
#     llm=None,  # Replace with OpenAI LLM client if applicable
#     tools=tools,
#     verbose=True,
#     agent_name="Socrates",
#     agent_role="Reflective Philosopher",
# )
#
#
# # Pydantic Model for incoming user messages
# class UserMessage(BaseModel):
#     user_input: str
#     context: Optional[Dict] = None  # Optional: Add context for richer interaction
#
#
# @app.get("/")
# def read_root():
#     return {"message": "Socrates Agent API is running."}
#
#
# @app.post("/interact")
# def interact_with_agent(user_message: UserMessage):
#     """
#     Endpoint to interact with the Socrates agent.
#
#     Args:
#         user_message (UserMessage): User input and optional context.
#
#     Returns:
#         JSON response containing the agent's response.
#     """
#     try:
#         # Process user input
#         agent.human_step(user_message.user_input)
#
#         # Generate response
#         response = agent.generate_response(user_message.user_input)
#
#         return {
#             "stage": agent.stage,
#             "response": response,
#             "memory": agent.memory,
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.post("/reset")
# def reset_agent():
#     """
#     Endpoint to reset the agent's memory and state.
#     """
#     try:
#         agent.memory = []
#         agent.stage = "Introduction"
#         return {"message": "Agent memory and stage reset."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
from partnerable_agent_with_memory import PartnerableAgent

app = FastAPI()

# Enable CORS for all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Socrates agent with SQLite memory (memory.db in the current directory)
agent = PartnerableAgent(db_path="memory.db", verbose=True)


class UserMessage(BaseModel):
    user_id: str
    user_input: str
    context: Optional[Dict] = None  # Optional additional context


@app.get("/")
def read_root():
    return {"message": "Socrates Agent API is running."}


# @app.post("/interact")
# def interact_with_agent(user_message: UserMessage):
#     try:
#         # Store user's message
#         agent.human_step(user_message.user_id, user_message.user_input)
#         # Generate the agent's response
#         response_text = agent.generate_response(user_message.user_id)
#         # Retrieve full conversation memory
#         conversation_text = agent.sqlite_memory.retrieve_memory(user_message.user_id)
#         return {
#             "response": response_text,
#             "memory": conversation_text
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
@app.post("/interact")
async def interact_with_agent(user_message: UserMessage):
    try:
        agent.human_step(user_message.user_id, user_message.user_input)
        response_text = agent.generate_response(user_message.user_id)
        conversation_text = agent.sqlite_memory.retrieve_memory(user_message.user_id)
        return {
            "response": response_text,
            "memory": conversation_text
        }
    except Exception as e:
        print("Exception:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset_agent():
    """
    This endpoint resets the local conversation memory of the agent.
    (Note: This does not delete the SQLite file. You may need to manually remove it.)
    """
    try:
        # For a simple reset, remove the memory file (or clear the tables)
        import os
        if os.path.exists("memory.db"):
            os.remove("memory.db")
        # Reinitialize the memory manager in the agent
        agent.sqlite_memory = agent.sqlite_memory.__class__(db_path="memory.db")
        agent.stage = "Introduction"
        return {"message": "Agent memory reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
