"""
Chat BOT
"""

from app.inference import rag_chain


def run():
    """function to run the chatbot in terminal."""
    print("Starting app...")

    # -------- Define Source
    index_to_use = input("Enter PDF name :")
    # --------

    print("\n****Chatbot is ready! Type `exit` or `quit` to stop.****\n")

    # Interactive chat loop
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break
        if not query.strip():
            print("Please enter a valid question.")
            continue

        # Get response from the RAG chain
        try:
            response = rag_chain(query, index_to_use)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    run()
