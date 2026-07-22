from concurrent.futures import ThreadPoolExecutor, as_completed

from config import get_model
from knowledge_agent import knowledge_agent
from compliance_agent import compliance_agent
from schedule_agent import schedule_agent
from supply_chain_agent import supply_chain_agent
from commissioning_agent import commissioning_agent


# =====================================================
# Model
# =====================================================

model = get_model()


# =====================================================
# Sub-Agents Consulted For A Full Report
# =====================================================

SECTION_AGENTS = {
    "Knowledge Summary": knowledge_agent,
    "Schedule Risk": schedule_agent,
    "Supply Chain": supply_chain_agent,
    "Compliance": compliance_agent,
    "Commissioning": commissioning_agent,
}


# =====================================================
# Run All Section Agents
# =====================================================

def gather_sections(query: str) -> dict:
    """
    Calls every specialist agent with the same query in parallel
    and collects their individual reports.

    Returns:
        dict mapping section title -> agent's raw answer (or an
        error message if that particular agent failed - one
        agent failing should not take down the whole report).
    """

    sections = {}

    with ThreadPoolExecutor(max_workers=len(SECTION_AGENTS)) as executor:

        future_to_section = {
            executor.submit(agent_fn, query): title
            for title, agent_fn in SECTION_AGENTS.items()
        }

        for future in as_completed(future_to_section):

            title = future_to_section[future]

            try:
                sections[title] = future.result()

            except Exception as error:
                sections[title] = (
                    f"[This section could not be generated: {error}]"
                )

    # Preserve a stable, readable order regardless of completion order
    ordered = {title: sections[title] for title in SECTION_AGENTS if title in sections}

    return ordered


# =====================================================
# Synthesize Executive Summary
# =====================================================

SUMMARY_PROMPT = """
You are an AI EPC Project Reporting Agent for Data Centre construction
projects.

Below are individual reports already produced by five specialist agents,
each analyzing the same user question from their own domain: Knowledge,
Schedule Risk, Supply Chain, Compliance, and Commissioning.

Your job is to write a short EXECUTIVE SUMMARY (5-8 bullet points) that:

1. Synthesizes ONLY what is stated across these reports below.
2. Highlights the most important findings, risks, and numbers.
3. Never invents information that isn't present in the reports.
4. If a section says information is unavailable, do not treat that as
   a finding - simply skip it or note the gap briefly.
5. Is concise, professional, and suitable for a project stakeholder
   who does not have time to read all five full reports.

Specialist Reports:

{combined_sections}

User Question:
{query}

Write the Executive Summary now.
"""


def build_executive_summary(query: str, sections: dict) -> str:

    combined_sections = "\n\n".join(
        f"----- {title} -----\n{content}"
        for title, content in sections.items()
    )

    prompt = SUMMARY_PROMPT.format(
        combined_sections=combined_sections,
        query=query,
    )

    response = model.generate_content(prompt)

    return response.text


# =====================================================
# Report Agent - Main Entry Point
# =====================================================

def report_agent(query: str) -> str:
    """
    Produces a single consolidated report by running every specialist
    agent against the same query and prepending an executive summary
    that ties their findings together.
    """

    sections = gather_sections(query)

    executive_summary = build_executive_summary(query, sections)

    report = "# EPC Project Consolidated Report\n\n"
    report += "## Executive Summary\n\n"
    report += executive_summary.strip() + "\n\n"

    for title, content in sections.items():
        report += f"## {title}\n\n"
        report += content.strip() + "\n\n"

    return report


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    while True:

        query = input("\nAsk for a Full Project Report (type exit to quit): ")

        if query.lower() == "exit":
            break

        answer = report_agent(query)

        print("\n" + "=" * 70)
        print(answer)
        print("=" * 70)
