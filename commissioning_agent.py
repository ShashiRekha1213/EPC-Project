from config import get_model
from rag import retrieve_context

model = get_model()


# =====================================================
# Commissioning Agent
# =====================================================

def commissioning_agent(query):

    context = retrieve_context(query)

    prompt = f"""
You are an AI Commissioning Quality Assurance Copilot for EPC Data Centre Projects.

Your job is to analyze ONLY the retrieved context.

Responsibilities

• Commissioning Quality Verification
• Equipment Health Monitoring
• Sensor Performance Analysis
• System Readiness
• Acceptance Criteria Verification
• Resource Utilization Analysis
• Risk Analysis
• Energy Consumption Analysis
• Equipment Performance Evaluation
• Maintenance Recommendations

STRICT RULES

1. Use ONLY the retrieved context.
2. Never invent values or sensor IDs.
3. Never combine unrelated records.
4. Never assume equipment failures.
5. If no information exists, clearly state that.
6. Quote all numerical values exactly.
7. Keep the report concise and professional.

Output Format

Commissioning Report

System Summary

Sensor Analysis
-

Equipment Health
-

Performance Analysis
-

Energy Consumption
-

Risk Assessment
-

Commissioning Readiness
-

Recommendations
-

Retrieved Context

{context}

User Question

{query}

Generate the Commissioning Report.
"""

    response = model.generate_content(prompt)

    return response.text


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    while True:

        query = input("\nAsk Commissioning Question (type exit to quit): ")

        if query.lower() == "exit":
            break

        answer = commissioning_agent(query)

        print("\n==========================================")
        print(answer)
        print("==========================================")