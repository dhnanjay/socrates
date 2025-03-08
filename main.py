from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
from partnerable_agent import PartnerableAgent
import uvicorn

# agent = PartnerableAgent(
#     llm=None,  # Replace with OpenAI LLM client if applicable
#     tools=tools,
#     verbose=True,
#     agent_name="Socrates",
#     agent_role="Reflective Philosopher",
# )

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins, all methods, all headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify specific domains like ["http://example.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mock Tools
def mock_tool(input_text: str) -> str:
    return f"Processed input: {input_text}"


tools = {"MockTool": mock_tool}

# Initialize Socrates agent
agent = PartnerableAgent(
    llm=None,  # Replace with OpenAI LLM client if applicable
    tools=tools,
    verbose=True,
    agent_name="Socrates",
    agent_role="Reflective Philosopher",
)


# Pydantic Model for incoming user messages
class UserMessage(BaseModel):
    user_input: str
    context: Optional[Dict] = None  # Optional: Add context for richer interaction


@app.get("/")
def read_root():
    return {"message": "Socrates Agent API is running."}


@app.post("/interact")
def interact_with_agent(user_message: UserMessage):
    """
    Endpoint to interact with the Socrates agent.

    Args:
        user_message (UserMessage): User input and optional context.

    Returns:
        JSON response containing the agent's response.
    """
    try:
        # Process user input
        agent.human_step(user_message.user_input)

        # Generate response
        response = agent.generate_response(user_message.user_input)

        return {
            "stage": agent.stage,
            "response": response,
            "memory": agent.memory,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset_agent():
    """
    Endpoint to reset the agent's memory and state.
    """
    try:
        agent.memory = []
        agent.stage = "Introduction"
        return {"message": "Agent memory and stage reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
