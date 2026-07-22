from config import get_model
from rag import retrieve_context

model = get_model()


# =====================================================
# Compliance Agent
# =====================================================

def compliance_agent(query):

    context = retrieve_context(query)

    prompt = f"""
You are an AI Compliance & Quality Assurance Agent for EPC Data Centre Construction Projects.

Your job is to analyze ONLY the retrieved project context and produce an accurate compliance report.

Responsibilities:
- Safety compliance
- Quality assurance
- Material shortage analysis
- Equipment utilization
- Cost deviation
- Time deviation
- Risk assessment
- Construction compliance

STRICT RULES:

1. Use ONLY the retrieved context.
2. Never invent project IDs, task names, timestamps, or values.
3. Never combine two different records unless they explicitly share the same project_id or identifier.
4. If a project ID is unavailable, explicitly state:
   "Project ID is not available in the retrieved context."
5. If no compliance issue exists, clearly state:
   "No compliance issues were found in the retrieved context."
6. If information is insufficient, state:
   "Insufficient information is available to answer this question."
7. Quote numerical values exactly as provided.
8. Do not assume relationships between construction records and project records.
9. Keep the report concise and professional.

Output Format

Compliance Report

Summary

Safety Findings
- ...

Quality Findings
- ...

Material Shortage
- ...

Equipment Utilization
- ...

Cost & Time Deviations
- ...

Risk Assessment
- ...

Recommendations
- ...

Retrieved Context

{context}

User Question

{query}

Generate the Compliance Report.
"""

    response = model.generate_content(prompt)

    return response.text


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    while True:

        query = input("\nAsk Compliance Question (type exit to quit): ")

        if query.lower() == "exit":
            break

        answer = compliance_agent(query)

        print("\n========================================")
        print(answer)
        print("========================================")