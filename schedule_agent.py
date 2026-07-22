from config import get_model
from rag import retrieve_context

model = get_model()


# =====================================================
# Schedule Risk Agent
# =====================================================

def schedule_agent(query):

    context = retrieve_context(query)

    prompt = f"""
You are an AI Schedule Risk Agent for EPC Data Centre Construction Projects.

Your responsibility is to analyze project schedules and identify potential risks using ONLY the retrieved context.

Responsibilities:
- Project progress analysis
- Schedule delay detection
- Critical path risk identification
- Workforce availability analysis
- Budget vs progress analysis
- Cost deviation analysis
- Time deviation analysis
- Resource utilization
- Delay mitigation suggestions

STRICT RULES:

1. Use ONLY the retrieved context.
2. Never invent project IDs, task names, dates, or values.
3. Never combine information from different records unless they explicitly share the same project_id.
4. If schedule information is unavailable, state:
   "Insufficient schedule information is available."
5. If no schedule risks are found, state:
   "No schedule risks were identified."
6. Quote numerical values exactly as they appear.
7. Keep the report professional.

Output Format

Schedule Risk Report

Project Summary

Progress Analysis
- ...

Schedule Risks
- ...

Critical Path Risks
- ...

Budget Analysis
- ...

Resource Analysis
- ...

Recommendations
- ...

Retrieved Context

{context}

User Question

{query}

Generate the Schedule Risk Report.
"""

    response = model.generate_content(prompt)

    return response.text


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    while True:

        query = input("\nAsk Schedule Question (type exit to quit): ")

        if query.lower() == "exit":
            break

        answer = schedule_agent(query)

        print("\n========================================")
        print(answer)
        print("========================================")