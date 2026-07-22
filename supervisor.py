"""
EPC AI Project - Supervisor Agent

Responsibilities:
    1. Understand the user's question.
    2. Select the most appropriate specialist agent using Gemini.
    3. Execute the selected specialist agent.
    4. Return a consistent response.

Architecture:

    User Query
         |
         v
    Supervisor LLM
         |
         v
    Agent Selection
         |
         +-------------------+
         |                   |
         v                   v
    Specialist Agent     Error Handler
         |
         v
    Final Response
"""

import json
import re
from typing import Callable, Dict, Optional

from config import get_model
from knowledge_agent import knowledge_agent
from compliance_agent import compliance_agent
from schedule_agent import schedule_agent
from supply_chain_agent import supply_chain_agent
from commissioning_agent import commissioning_agent


# ============================================================
# Configuration
# ============================================================

VALID_AGENTS = {
    "knowledge",
    "compliance",
    "schedule",
    "supply_chain",
    "commissioning",
}


# ============================================================
# Router Model (uses the same Gemini setup as every other agent,
# configured once in config.py)
# ============================================================

router_model = get_model()


# ============================================================
# Agent Registry
# ============================================================

AGENT_REGISTRY: Dict[str, Callable] = {
    "knowledge": knowledge_agent,
    "compliance": compliance_agent,
    "schedule": schedule_agent,
    "supply_chain": supply_chain_agent,
    "commissioning": commissioning_agent,
}


# ============================================================
# Agent Descriptions
# ============================================================

AGENT_DESCRIPTIONS = {

    "knowledge": """
        Handles general EPC project knowledge and information.
        Use this agent for questions about project details,
        project status, project data, project records,
        budgets, locations, engineers, and general project
        information.
    """,

    "compliance": """
        Handles EPC compliance, quality assurance,
        quality control, safety, construction standards,
        material issues, and compliance-related analysis.
    """,

    "schedule": """
        Handles project planning and scheduling.
        Use this agent for project timelines, delays,
        completion dates, progress, critical paths,
        schedule risks, and resource-related schedule issues.
    """,

    "supply_chain": """
        Handles procurement and supply chain operations.
        Use this agent for suppliers, vendors, tenders,
        procurement, purchasing, contracts, materials,
        deliveries, and supply-related issues.
    """,

    "commissioning": """
        Handles commissioning activities.
        Use this agent for equipment readiness,
        system readiness, commissioning status,
        equipment performance, sensors, testing,
        and commissioning-related issues.
    """,

}


# ============================================================
# Build Supervisor Prompt
# ============================================================

def build_supervisor_prompt(query: str) -> str:
    """
    Creates the prompt used by Gemini to select
    the most appropriate EPC specialist agent.
    """

    agent_information = "\n".join(
        f"""
        Agent: {agent_name}
        Description:
        {description.strip()}
        """
        for agent_name, description
        in AGENT_DESCRIPTIONS.items()
    )

    return f"""
You are the Supervisor Agent for an EPC
(Engineering, Procurement, and Construction)
Project Management AI system.

Your task is to understand the user's question
and select the ONE specialist agent that is
best qualified to answer it.

Do not select an agent based on simple keyword matching.
Understand the meaning, context, and intent of the question.

Available specialist agents:

{agent_information}

User Question:
----------------
{query}
----------------

Return ONLY valid JSON.

Required format:

{{
    "agent": "agent_name"
}}

The value of "agent" MUST be exactly one of:

- knowledge
- compliance
- schedule
- supply_chain
- commissioning
"""


# ============================================================
# Extract JSON from LLM Response
# ============================================================

def extract_json(response_text: str) -> dict:
    """
    Safely extracts a JSON object from the Gemini response.

    Handles cases where the model returns JSON inside
    Markdown code fences.
    """

    if not response_text:
        raise ValueError(
            "Supervisor returned an empty response."
        )

    response_text = response_text.strip()

    # Remove Markdown code fences if present
    response_text = re.sub(
        r"```json\s*",
        "",
        response_text,
        flags=re.IGNORECASE
    )

    response_text = re.sub(
        r"```\s*$",
        "",
        response_text
    )

    response_text = response_text.strip()

    try:

        return json.loads(response_text)

    except json.JSONDecodeError:

        # Try to find the first JSON object
        match = re.search(
            r"\{.*\}",
            response_text,
            re.DOTALL
        )

        if not match:
            raise ValueError(
                "Supervisor returned invalid JSON."
            )

        return json.loads(
            match.group(0)
        )


# ============================================================
# Select Specialist Agent
# ============================================================

def select_agent(query: str) -> str:
    """
    Uses Gemini to understand the user's question
    and select the appropriate specialist agent.

    Returns:
        str: Selected agent name.
    """

    prompt = build_supervisor_prompt(query)

    response = router_model.generate_content(prompt)

    result = extract_json(response.text)

    selected_agent = result.get("agent")

    if selected_agent not in VALID_AGENTS:

        raise ValueError(
            f"Invalid agent selected by supervisor: "
            f"{selected_agent}"
        )

    return selected_agent


# ============================================================
# Execute Specialist Agent
# ============================================================

def execute_agent(agent_name: str, query: str):
    """
    Executes the selected specialist agent.
    """

    agent_function = AGENT_REGISTRY.get(agent_name)

    if agent_function is None:

        raise ValueError(
            f"No agent registered for: "
            f"{agent_name}"
        )

    return agent_function(query)


# ============================================================
# Main Supervisor Function
# ============================================================

def supervisor(query: str, agent: Optional[str] = None) -> dict:
    """
    Main entry point for the EPC AI Supervisor.

    Parameters:
        query:
            User's EPC-related question.

        agent:
            Optional manually selected agent.

            If None:
                Gemini automatically selects the agent.

            If provided:
                The selected agent is called directly, skipping
                the LLM routing step.

    Returns:
        Dictionary containing:

        {
            "success": True,
            "agent": "schedule",
            "answer": "..."
        }
    """

    # --------------------------------------------------------
    # Validate User Query
    # --------------------------------------------------------

    if not query or not query.strip():

        return {
            "success": False,
            "agent": "supervisor",
            "answer": "Please provide a valid question.",
            "error": "Empty query.",
        }

    query = query.strip()

    try:

        # ----------------------------------------------------
        # Agent Selection
        # ----------------------------------------------------

        if agent:

            agent = agent.strip().lower()

            if agent not in VALID_AGENTS:

                return {
                    "success": False,
                    "agent": "supervisor",
                    "answer": "The selected agent is invalid.",
                    "error": f"Unknown agent: {agent}",
                }

            selected_agent = agent

        else:

            selected_agent = select_agent(query)

        # ----------------------------------------------------
        # Execute Agent
        # ----------------------------------------------------

        answer = execute_agent(selected_agent, query)

        # ----------------------------------------------------
        # Return Successful Response
        # ----------------------------------------------------

        return {
            "success": True,
            "agent": selected_agent,
            "answer": answer,
        }

    except Exception as error:

        # ----------------------------------------------------
        # Error Handling
        # ----------------------------------------------------

        print("Supervisor Error:", str(error))

        return {
            "success": False,
            "agent": agent or "supervisor",
            "answer": (
                "I was unable to process your request. "
                "Please try again."
            ),
            "error": str(error),
        }


# ============================================================
# Command Line Testing
# ============================================================

if __name__ == "__main__":

    print("\n==========================================")
    print("       EPC AI SUPERVISOR")
    print("==========================================")

    print("\nAvailable Agents:")

    for agent_name in AGENT_REGISTRY:
        print(f"  - {agent_name}")

    print("\nType 'exit' to stop.")

    while True:

        try:

            query = input("\nUser Question: ").strip()

            if query.lower() == "exit":
                print("\nSupervisor stopped.")
                break

            result = supervisor(query)

            print("\n------------------------------------------")
            print("Selected Agent:", result.get("agent"))
            print("------------------------------------------")
            print(result.get("answer"))
            print("------------------------------------------")

        except KeyboardInterrupt:

            print("\n\nSupervisor stopped.")
            break
