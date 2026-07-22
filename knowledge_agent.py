from config import get_model
from rag import retrieve_context

model = get_model()


# =====================================================
# Knowledge Agent
# =====================================================

def knowledge_agent(query):

    context = retrieve_context(query)

    prompt = f"""
You are an AI EPC Project Knowledge Agent.

Your responsibility is to answer questions related to:

• Project information
• Project status
• Tasks
• Budget
• Cost
• Progress
• Assigned engineers
• Locations
• Project timelines

Instructions:

1. Answer ONLY from the retrieved context.
2. Never make assumptions.
3. If information is unavailable, clearly state:
   "The requested information is not available in the project database."
4. Explain clearly.
5. Use bullet points whenever suitable.
6. Mention important project values if available.
7. Keep the response professional.

Retrieved Context:

{context}

User Question:

{query}

Answer:
"""

    response = model.generate_content(prompt)

    return response.text


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    while True:

        query = input("\nAsk Project Question (type exit to quit): ")

        if query.lower() == "exit":
            break

        answer = knowledge_agent(query)

        print("\n==============================")
        print(answer)
        print("==============================")