import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq


class Chatbot:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.model = ChatGroq(model="Gemma2-9b-It", groq_api_key=self.groq_api_key)

        # Initialize message store
        self.store = {}

        # Setup prompt template with system message
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Create the chain
        self.chain = self.prompt | self.model

        # Setup message history
        self.with_message_history = RunnableWithMessageHistory(
            self.chain, self.get_session_history, input_messages_key="messages"
        )

        # Setup message trimmer
        self.trimmer = trim_messages(
            max_tokens=45,
            strategy="last",
            token_counter=self.model,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

        # Setup final chain with trimmer
        self.final_chain = (
            RunnablePassthrough.assign(messages=itemgetter("messages") | self.trimmer)
            | self.prompt
            | self.model
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def chat(self, message: str, session_id: str, language: str = "English") -> str:
        """Send a message and get a response"""
        config = {"configurable": {"session_id": session_id}}

        response = self.with_message_history.invoke(
            {"messages": [HumanMessage(content=message)], "language": language},
            config=config,
        )

        return response.content

    def chat_with_history(
        self, message: str, history: list, language: str = "English"
    ) -> str:
        """Chat with custom history"""
        response = self.final_chain.invoke(
            {
                "messages": history + [HumanMessage(content=message)],
                "language": language,
            }
        )
        return response.content


def main():
    # Initialize chatbot
    chatbot = Chatbot()

    # Example usage with session management
    print("Chat Example 1 (with session management):")
    session_id = "chat1"

    # First message
    response = chatbot.chat("Hi, My name is Krish", session_id)
    print("User: Hi, My name is Krish")
    print("Bot:", response)

    # Second message
    response = chatbot.chat("What's my name?", session_id)
    print("\nUser: What's my name?")
    print("Bot:", response)

    # Example with different language
    print("\nChat Example 2 (in Hindi):")
    session_id = "chat2"
    response = chatbot.chat("Hi, My name is Krish", session_id, language="Hindi")
    print("User: Hi, My name is Krish")
    print("Bot:", response)

    # Example with custom history
    print("\nChat Example 3 (with custom history):")
    history = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
    ]

    response = chatbot.chat_with_history("What ice cream do I like?", history)
    print("User: What ice cream do I like?")
    print("Bot:", response)


if __name__ == "__main__":
    main()
