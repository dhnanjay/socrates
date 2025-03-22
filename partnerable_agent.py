import os
import json
from openai import OpenAI
import logging
from typing import Dict, Any, List

# Set the logging level to WARNING or ERROR to suppress debug logs
logging.basicConfig(level=logging.WARNING)

# Read tokens from file
config_path = "config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    openai_api_key = config.get("OPENAI_API_KEY", "")
else:
    openai_api_key = ""


client = OpenAI(api_key=openai_api_key)


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate the cost of a request based on input and output tokens."""
    if model == "gpt-4o":
        input_cost_per_1k = 0.03  # $0.03 per 1,000 input tokens
        output_cost_per_1k = 0.06  # $0.06 per 1,000 output tokens
    elif model == "gpt-3.5-turbo":
        input_cost_per_1k = 0.0015  # $0.0015 per 1,000 input tokens
        output_cost_per_1k = 0.002  # $0.002 per 1,000 output tokens
    else:
        raise ValueError(f"Unsupported model: {model}")

    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost
    return total_cost


def call_llm(prompt: str, system_prompt: str, model="gpt-4o", client=client) -> Dict[str, Any]:
    """Call the OpenAI chat completion API for normal responses."""
    try:
        # Build the messages array
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Make OpenAI API call
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        # Extract response details
        response_text = response.choices[0].message.content.strip()
        prompt_tokens = response.usage.prompt_tokens  # Input tokens
        completion_tokens = response.usage.completion_tokens  # Output tokens
        total_tokens = response.usage.total_tokens  # Total tokens

        # Calculate cost using input and output tokens
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


def call_llm_json(
        prompt: str,
        system_prompt: str,
        model="gpt-4o",
        json_schema=None,
        client=client
) -> Dict[str, Any]:
    """Call the OpenAI chat completion API and enforce JSON schema response."""
    try:
        if not json_schema:
            raise ValueError("A valid JSON schema must be provided.")

        # Build the messages array
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Make OpenAI API call with JSON schema enforcement
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "json_output",  # Name of the schema
                    "schema": json_schema
                }
            }
        )

        # Extract response details
        json_response = response.choices[0].message.content.strip()
        prompt_tokens = response.usage.prompt_tokens  # Input tokens
        completion_tokens = response.usage.completion_tokens  # Output tokens
        total_tokens = response.usage.total_tokens  # Total tokens

        # Calculate cost
        total_cost = calculate_cost(prompt_tokens, completion_tokens, model)

        return {
            "json_response": json_response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": total_cost
        }

    except Exception as e:
        logging.error(f"Error during JSON schema chat completion: {e}")
        return {
            "json_response": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0
        }


class PartnerableAgent:
    """An AI agent built on the principles of 'Partnerable' with a Socratic, principle-driven approach."""

    def __init__(self, llm, verbose: bool = False, **kwargs):
        self.llm = llm
        self.verbose = verbose
        self.agent_name = kwargs.get("agent_name", "Socrates")
        self.agent_role = kwargs.get("agent_role", "Representative Philosopher")
        self.memory = []
        self.stage = "Introduction"
        self.tools = kwargs.get("tools", {})
        self.core_principles = self.initialize_core_principles()

    def initialize_core_principles(self) -> str:
        """Define the principles as part of the agent's knowledge base."""
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

    def seed_agent(self):
        """Initialize the agent with an introduction."""
        intro = (
            f"Hello! I am {self.agent_name}, your reflective guide. Let's explore your challenges "
            "and collaboratively find solutions. How can I assist you today?"
        )
        self.memory.append({"agent": intro})
        if self.verbose:
            print(f"[{self.agent_name}] Agent initialized.")
        return intro

    def determine_conversation_stage(self, user_input: str = "") -> str:
        """
        Determine the stage of interaction based on user input and history using an LLM API call.
        The LLM analyzes the context and provides the appropriate conversation stage.
        """
        try:
            # Create a context string from the memory and user input
            memory_summary = "\n".join(
                [
                    f"User: {interaction.get('user', 'N/A')}, Agent: {interaction.get('agent', 'N/A')}"
                    for interaction in self.memory[-5:]
                ]
            )
            context = f"""
          You are a conversation stage analyzer for an AI agent. The agent follows the principles of 'Partnerable' and is assisting a user with reflective guidance.
          Your task is to determine the current stage of the conversation based on the input and recent interactions.
          Stages to consider:
          1. Exploration: The user is seeking to understand or explore a problem, situation, or topic.
          2. Alignment: The user is attempting to align multiple perspectives or resolve a conflict.
          3. Ownership: The user is focusing on responsibility, accountability, or taking charge of a task or decision.
          4. Leadership: The user is working to lead, facilitate, or manage others to achieve a goal.

          Recent Conversation History (up to 5 interactions):
          {memory_summary}

          User Input:
          {user_input}

          Which stage does this conversation most likely correspond to? Provide only the stage name (e.g., "Exploration").
          """

            # Call the LLM to determine the stage
            response = call_llm(
                prompt=context,
                system_prompt="You are a specialized stage analyzer for conversational interactions.",
                model="gpt-4o"
            )

            # Extract the stage from the response
            suggested_stage = response["response_text"].strip()

            # Validate the stage
            valid_stages = ["Exploration", "Alignment", "Ownership", "Leadership"]
            if suggested_stage in valid_stages:
                self.stage = suggested_stage
            else:
                self.stage = "Exploration"  # Default to Exploration if the response is unclear

            if self.verbose:
                print(f"[{self.agent_name}] Determined Conversation Stage: {self.stage}")
            return self.stage

        except Exception as e:
            logging.error(f"Error determining conversation stage: {e}")
            self.stage = "Exploration"  # Default stage in case of failure
            return self.stage

    def actionable_instructions(self, stage: str) -> str:
        """Provide specific actionable instructions for each stage."""
        instructions = {
            "Exploration": """
            Actions:
            - Encourage the user to articulate their thoughts or questions.
            - Ask reflective questions to help them clarify their goals.
            - Provide open-ended prompts for deeper exploration.
            """,
            "Alignment": """
            Actions:
            - Facilitate open dialogue to harmonize different perspectives.
            - Summarize inputs to ensure alignment.
            - Propose mutually beneficial adjustments to plans.
            """,
            "Ownership": """
            Actions:
            - Take clear accountability for specific tasks or outcomes.
            - Communicate transparently about responsibilities and criteria for success.
            - Regularly update stakeholders on progress and results.
            """,
            "Leadership": """
            Actions:
            - Balance diverse inputs and lead collaborative decision-making.
            - Create a 'safe space' for conflicting ideas to surface.
            - Make informed decisions that balance immediate needs with long-term goals.
            """
        }
        return instructions.get(stage, "No specific actions for this stage.")

    def generate_response(self, user_input: str) -> str:
        """Generate a response based on the conversation stage, memory, and principles."""
        prompt = user_input  # self.create_prompt(user_input)
        system_prompt = self.create_system_prompt()
        try:
            # Use the custom `call_llm` function for generating responses
            response = call_llm(
                prompt=prompt,
                system_prompt=f"You are {self.agent_name}, guided by the principles of 'Partnerable'. " + system_prompt,
                model="gpt-4o"  # You can adjust the model here
            )
            return response["response_text"]
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error generating my response."

    ######
    def create_system_prompt(self) -> str:
        """Create a detailed, stage-specific prompt for the Socrates agent that includes:
        - Memory of the last N interactions, clearly segmented.
        - Clear identification of the agent's role and core principles.
        - Current conversation stage and actionable instructions for that stage.
        - Guidance to conclude interactions based on context.
        """
        # Construct the memory summary (last 5 interactions)
        memory_summary = "\n".join(
            [
                f"User: {interaction.get('user', 'N/A')}\nAgent: {interaction.get('agent', 'N/A')}"
                for interaction in self.memory[-5:]
            ]
        )

        # Get actionable instructions for the current stage
        actionable_stage_instructions = self.actionable_instructions(self.stage)

        # System/Developer prompt
        system_prompt = f"""
        Your purpose is to assist the user in exploring challenges, fostering understanding, and discovering actionable insights through thoughtful dialogue and questioning.
         Your approach combines reflective inquiry with guidance grounded in the following principles:

      ---

      ### Core Principles:
      {self.core_principles}
      Strictly follow the core principles

      ---
      ### Socratic Questioning Principles:
      - Always aim to clarify the user's underlying assumptions.
      - You may Use probing questions to deepen understanding:
        - What do you mean by that?
        - Why do you think this approach might work?
        - What evidence supports this perspective?
    - Encourage self-examination and critical thinking which is what Socrates advocated:
    - Are there alternative viewpoints to consider?
    - What might be the consequences of this choice?
    - How does this align with your long-term goals?
    - Basically, The answer user seek lies not in external solutions but within their own ability to critically examine underlying assumptions, question their beliefs,
     and challenge the boundaries of their current thinking
      ---

      ### The Current Conversation Context is :
      Conversation Stage: {self.stage}

      ---

      ### Based on the Conversation Stage use the following Stage-Specific Instructions to respond to the user:
      {actionable_stage_instructions}

      ---
      So that you have good historical context with the user, below is the past interaction history:
      Memory of Last 5 Interactions** (most recent first):
      {memory_summary}

      ---

      ### Guidelines for Interaction:
      - Respond thoughtfully, adhering to the core principles and stage-specific instructions.
      - Incorporate the user's input and past interactions when formulating your response.
      - Conclude the interaction when appropriate, ensuring the user feels supported and has actionable next steps.
      When using Socratic questioning and Core principles:
	    1.	Stay neutral and curious—don’t impose your own beliefs or judgments.
	    2.	Ask open-ended questions that encourage thought rather than yes/no answers.
	    3.	Follow up on responses with deeper questions to encourage reflection and exploration.
	    4.	Create a safe and respectful environment where the person feels comfortable sharing and reflecting.


      """

        # Return the full prompt
        return system_prompt

    ####3

    def create_prompt(self, user_input: str) -> str:
        """Create a stage-specific prompt that incorporates memory and principles."""
        memory_summary = "\n".join(
            [
                f"User: {interaction.get('user', 'N/A')}, Agent: {interaction.get('agent', 'N/A')}"
                for interaction in self.memory[-5:]
            ]
        )
        actionable_stage_instructions = self.actionable_instructions(self.stage)
        return f"""
  Conversation Stage: {self.stage}

  Memory Summary (Last 5 Interactions):
  {memory_summary}

  User Input: {user_input}

  {self.core_principles}

  Actionable Instructions for this Stage:
  {actionable_stage_instructions}

  Respond thoughtfully, adhering to the principles and stage-specific instructions.
  """

    def step(self):
        """Simulate the agent's response step."""
        last_user_input = self.memory[-1]["user"] if self.memory else ""
        response = self.generate_response(last_user_input)
        self.memory.append({"agent": response})
        print(f"{self.agent_name}: {response}")

    def human_step(self, user_input: str):
        """Record user input and update the conversation stage."""
        self.memory.append({"user": user_input})
        self.determine_conversation_stage(user_input)

    def use_tool(self, tool_name: str, tool_input: str) -> str:
        """Use a specified tool, if available."""
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            result = tool(tool_input)
            if self.verbose:
                print(f"[{self.agent_name}] Tool '{tool_name}' used with input: {tool_input}")
            return result
        else:
            raise ValueError(f"Tool '{tool_name}' not available.")

    def adaptive_learning(self, feedback: str):
        """Adaptively learn and refine understanding based on user feedback."""
        self.memory.append({"feedback": feedback})
        if self.verbose:
            print(f"[{self.agent_name}] Feedback received and stored.")
