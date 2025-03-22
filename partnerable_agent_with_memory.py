import os
import json
import logging
from typing import Dict, Any, List
from openai import OpenAI
from sqlite_memory_manager import SQLiteMemory

# Configure logging (set to WARNING to suppress debug logs)
logging.basicConfig(level=logging.WARNING)

# Load OpenAI API key from config.json (or environment)
config_path = "config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    openai_api_key = config.get("OPENAI_API_KEY", "")
else:
    openai_api_key = ""

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    if model == "gpt-4o":
        input_cost_per_1k = 0.03
        output_cost_per_1k = 0.06
    elif model == "gpt-3.5-turbo":
        input_cost_per_1k = 0.0015
        output_cost_per_1k = 0.002
    else:
        raise ValueError(f"Unsupported model: {model}")
    return (input_tokens / 1000) * input_cost_per_1k + (output_tokens / 1000) * output_cost_per_1k


def call_llm(messages: List[Dict[str, str]], model="gpt-4o", client=client) -> Dict[str, Any]:
    """
    Call the OpenAI chat completion API with a list of messages.
    Each message is a dict with "role" and "content".
    """
    try:
        response = client.chat.completions.create(model=model, messages=messages)
        response_text = response.choices[0].message.content.strip()
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        total_cost = calculate_cost(prompt_tokens, completion_tokens, model)
        return {
            "response_text": response_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": total_cost
        }
    except Exception as e:
        logging.error(f"Error during chat completion: {e}")
        return {
            "response_text": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0
        }


class PartnerableAgent:
    """
    An AI agent built on 'Partnerable' principles that uses SQLite for persistent conversation memory.
    It retrieves the rolling memory (latest summary + last n-1 interactions) and uses that as context
    to determine the current conversation stage and generate responses.
    """

    def __init__(self, db_path="memory.db", verbose: bool = False, **kwargs):
        self.verbose = verbose
        self.agent_name = kwargs.get("agent_name", "Socrates")
        self.agent_role = kwargs.get("agent_role", "Representative Philosopher")
        self.tools = kwargs.get("tools", {})
        self.stage = "Introduction"
        self.sqlite_memory = SQLiteMemory(db_path=db_path)
        self.core_principles = self.initialize_core_principles()

    def initialize_core_principles(self) -> str:
        return """
        Core Principles:
        1. Representative Behavior:
            Act on behalf of users while managing visible and absent expectations.
            Questions to ask:
                - Whose interests am I representing?
                - How does this decision align with their needs?
        2. Center |Space:
            Balance independence with collaboration, creating safe spaces for dialogue.
            Actions to take:
                - Encourage open-ended questions.
                - Validate diverse perspectives.
                - Mediate conflicts to find common ground.
        3. Ownership:
            Take clear accountability for actions and ensure transparency.
            Statements to make:
                - "I am responsible for [specific outcome]."
                - "Here are the criteria for success, and I will keep you updated."
        4. Aligning:
            Harmonize diverse perspectives to create mutually beneficial outcomes.
            Reflection prompts:
                - How can I adjust this plan to better serve everyone involved?
                - What compromises or adjustments are needed to achieve alignment?
        5. 	•	Core Principle as an essence of Partnerable:
Representing is not merely about being physically present but about actively caring for and advocating on behalf of others—even when they are not directly in front of us. It’s a dynamic, ongoing practice of aligning actions with the interests and needs of both those we represent and those who represent us.
	•	Intentional Attention & Routine:
The process requires consciously managing our attention. Since the human brain naturally focuses on what is present, we must intentionally create routines (e.g., daily reflections, regular reviews of key relationships) to keep the interests of absent stakeholders in active focus. This is less about managing time and more about managing intention and attention.
	•	Continuous Update & Alignment:
Every interaction is a chance to update and recalibrate priorities. By continually revisiting the interests of both parties, we ensure that our actions remain aligned with evolving needs and expectations.
	•	Inner and Interpersonal Practices:
	•	Inner Routines: Techniques like a pre-meeting “check” (identifying who is absent but impacted) and periodic reviews of key relationships help make these interests top-of-mind.
	•	Interpersonal Routines: Practices such as active and empathetic listening (Socratic listening), clear signaling of our own perspectives, and structured communication (clarifying scope, purpose, and timelines) create a safe space for mutual understanding and effective collaboration.
	•	Organized Collaboration:
Just as cells work in unison to form an organ, our individual efforts, when organized deliberately (through ownership arenas, status updates, and clear handoffs), contribute to a coherent and effective partnership. This ensures that all stakeholders’ needs are addressed and that any conflicts are managed through clear, structured communication channels.
	•	Outcome:
By embedding these routines and practices into daily interactions, we build a “partnerable” approach that prioritizes accountability, empathy, and proactive management of both visible and less visible interests. This method cultivates trust and ensures that the representation of interests is continuously maintained, even when individuals are not directly present.
        """

    def seed_agent(self, user_id: str) -> str:
        """
        Optionally seeds the conversation with an introductory message.
        """
        intro = (
            f"Hello! I am {self.agent_name}, your reflective guide. "
            "Let's explore your challenges and find solutions together. How can I help you today?"
        )
        self.sqlite_memory.store_message(user_id, intro, role="agent")
        if self.verbose:
            print(f"[{self.agent_name}] Seeded conversation for {user_id}.")
        return intro

    def human_step(self, user_id: str, user_input: str):
        """
        Stores the user's input in SQLite. If no conversation history exists, seeds the conversation.
        Also triggers a stage determination.
        """
        # If conversation is empty, seed the agent.
        if not self.sqlite_memory.retrieve_memory(user_id).strip():
            self.seed_agent(user_id)
        self.sqlite_memory.store_message(user_id, user_input, role="user")
        # Update conversation stage based on current history.
        # self.determine_conversation_stage(user_id)

    def determine_conversation_stage(self, user_id: str) -> str:
        """
        Retrieves the current conversation context from SQLite and uses the LLM to determine
        the conversation stage. Expected stages: Exploration, Alignment, Ownership, or Leadership.
        """
        conversation_text = self.sqlite_memory.retrieve_memory(user_id)
        prompt = (
            "You are a conversation stage analyzer for an AI agent. "
            "Based on the following conversation history, determine the current conversation stage. "
            "Stages to consider:"
            "1. Exploration: The user is seeking to understand or explore a problem, situation, or topic."
            "2. Alignment: The user is attempting to align multiple perspectives or resolve a conflict."
            "3. Ownership: The user is focusing on responsibility, accountability or taking charge of a task or decision."
            "4. Leadership: The user is working to lead, facilitate, or manage others to achieve a goal.\n\n"
            f"Conversation History:\n{conversation_text}\n\n"
            "Return only the stage name."
        )
        messages = [
            {"role": "system", "content": "You are a conversation stage analyzer."},
            {"role": "user", "content": prompt}
        ]
        response = call_llm(messages, model="gpt-4o")
        suggested_stage = (response["response_text"] or "").strip()
        valid_stages = ["Exploration", "Alignment", "Ownership", "Leadership"]
        if suggested_stage not in valid_stages:
            suggested_stage = "Exploration"
        self.stage = suggested_stage
        if self.verbose:
            print(f"[{self.agent_name}] Determined stage: {self.stage}")
        return self.stage

    def actionable_instructions(self, stage: str) -> str:
        """
        Returns actionable instructions based on the current conversation stage.
        """
        instructions = {
            "Exploration": (
                "Encourage the user to explain more about their situation and feelings. "
                "Ask open-ended questions to uncover deeper details."
            ),
            "Alignment": (
                "Help the user reconcile different perspectives. "
                "Ask clarifying questions that connect ideas and point out commonalities."
            ),
            "Ownership": (
                "Guide the user to take responsibility for decisions. "
                "Ask what actions they can take and how they will hold themselves accountable."
            ),
            "Leadership": (
                "Encourage the user to take initiative and guide others. "
                "Ask how they would lead the process and what steps they envision."
            )
        }
        return instructions.get(stage, "")

    def create_system_prompt(self, user_id: str) -> str:
        """
        Create a detailed system prompt for the LLM that includes:
          - The agent's core principles.
          - The current conversation stage.
          - Actionable instructions for that stage.
          - The persistent conversation history.
        """
        current_stage = self.determine_conversation_stage(user_id)
        instructions = self.actionable_instructions(current_stage)
        conversation_history = self.sqlite_memory.retrieve_memory(user_id)
        system_prompt = (
            f"You are {self.agent_name}, a reflective and principle-driven agent. Your role is to guide the user through "
            f"challenging conversations based on the following core principles:\n\n{self.core_principles}\n\n"
            f"Current Conversation Stage: {current_stage}\n"
            f"Actionable Instructions: {instructions}\n\n"
            f"Conversation History:\n{conversation_history}\n\n"
            "Based on the above, respond thoughtfully to the user's latest input."
            """"### Guidelines for Interaction:
            - Respond thoughtfully, adhering to the core principles and stage-specific instructions.
        - Incorporate the user's input and past interactions when formulating your response.
        - Conclude the interaction when appropriate, ensuring the user feels supported and has actionable next steps.
        When using Socratic questioning and Core principles:
        1.    Stay neutral and curious—don’t impose your own beliefs or judgments.
        2.    Follow up on responses with deeper questions to encourage reflection and exploration.
        3.    Create a safe and respectful environment where the person feels comfortable sharing and reflecting."""
        )
        return system_prompt

    def _build_messages_for_llm(self, conversation_text: str, system_prompt: str) -> List[Dict[str, str]]:
        """
        Converts the conversation history (including summary and unsummarized messages) into a list
        of message dictionaries suitable for the LLM API. It also prepends the detailed system prompt.
        """
        messages = [{"role": "system", "content": system_prompt}]
        for line in conversation_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("SUMMARY:"):
                content = line[len("SUMMARY:"):].strip()
                messages.append({"role": "system", "content": f"Conversation Summary: {content}"})
            elif line.startswith("USER:"):
                content = line[len("USER:"):].strip()
                messages.append({"role": "user", "content": content})
            elif line.startswith("AGENT:"):
                content = line[len("AGENT:"):].strip()
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "system", "content": line})
        return messages

    # def generate_response(self, user_id: str, user_input: str) -> str:
    #     """
    #     Combines the persistent conversation context with a fresh user input to generate the agent's response.
    #     The system prompt is built dynamically (which includes asynchronous summarization via SQLiteMemory).
    #     """
    #     # First, store the latest user input
    #     self.sqlite_memory.store_message(user_id, user_input, role="user")
    #     # Build a system prompt that includes the conversation history and stage instructions
    #     system_prompt = self.create_system_prompt(user_id)
    #     # Retrieve the conversation history after storing the new input
    #     conversation_text = self.sqlite_memory.retrieve_memory(user_id)
    #     # Build messages for LLM: system prompt + conversation lines
    #     messages = self._build_messages_for_llm(conversation_text, system_prompt)
    #     # Call the LLM to generate a response
    #     llm_response = call_llm(messages, model="gpt-4o")
    #     agent_text = llm_response["response_text"] or "I'm sorry, I have no response."
    #     # Store the agent's response asynchronously
    #     self.sqlite_memory.store_message(user_id, agent_text, role="agent")
    #     if self.verbose:
    #         print(f"[{self.agent_name}] Generated response for {user_id}: {agent_text}")
    #     return agent_text

    def generate_response(self, user_id: str) -> str:
        """
        Generates a response based on the persistent conversation history.
        It uses the system prompt (including current stage and instructions) and the conversation history.
        The agent's response is stored in SQLite and returned.
        """
        system_prompt = self.create_system_prompt(user_id)
        conversation_text = self.sqlite_memory.retrieve_memory(user_id)
        messages = self._build_messages_for_llm(conversation_text, system_prompt)
        llm_response = call_llm(messages, model="gpt-4o")
        agent_text = llm_response["response_text"] or "I'm sorry, I have no response."
        self.sqlite_memory.store_message(user_id, agent_text, role="agent")
        if self.verbose:
            print(f"[{self.agent_name}] Generated response for {user_id}: {agent_text}")
        return agent_text

    def use_tool(self, user_id: str, tool_name: str, tool_input: str) -> str:
        """
        Use a specified tool (if available) and store the usage in memory.
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not available.")
        result = self.tools[tool_name](tool_input)
        self.sqlite_memory.store_message(user_id, f"Used tool '{tool_name}' with input '{tool_input}'.", role="agent")
        return result

    def adaptive_learning(self, user_id: str, feedback: str):
        """
        Store user feedback in the persistent memory.
        """
        self.sqlite_memory.store_message(user_id, feedback, role="user")
        if self.verbose:
            print(f"[{self.agent_name}] Stored feedback for {user_id}.")
