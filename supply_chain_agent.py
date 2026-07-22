from config import get_model
from rag import retrieve_context

model = get_model()


# =====================================================
# Supply Chain Agent
# =====================================================

def supply_chain_agent(query):

    context = retrieve_context(query)

    prompt = f"""
You are an AI Supply Chain Risk Agent for EPC Data Centre Construction Projects.

Analyze ONLY the retrieved context.

Your responsibilities include:

• Procurement monitoring
• Tender analysis
• Vendor monitoring
• Delivery tracking
• Procurement risks
• Contract award analysis
• Deadline monitoring
• Country-wise procurement
• Procurement bottlenecks
• Supply chain recommendations

STRICT RULES

1. Use ONLY the retrieved context.
2. Never invent procurement records.
3. Never invent vendors, countries or project IDs.
4. Never combine unrelated records.
5. If information is unavailable, clearly state that.
6. Quote numerical values exactly.
7. Keep the report concise and professional.

Output Format

Supply Chain Report

Summary

Procurement Status
-

Tender Analysis
-

Delivery Risks
-

Country Analysis
-

Contract Awards
-

Recommendations
-

Retrieved Context

{context}

User Question

{query}

Generate the report.
"""

    response = model.generate_content(prompt)

    return response.text


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    while True:

        query = input("\nAsk Supply Chain Question (type exit to quit): ")

        if query.lower() == "exit":
            break

        answer = supply_chain_agent(query)

        print("\n==========================================")
        print(answer)
        print("==========================================")