from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

load_dotenv()

# chatprompttemplate
prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")

messages = prompt.format_messages(adjective="funny", topic="cats")

# print(messages)

def demo_basic_templates():
    """Basic ChatPromptTemplate usage."""

    # Simple template
    simple = ChatPromptTemplate.from_template("Translate '{text}' to {language}")

    messages = simple.format_messages(text="Hello, world!", language="French")
    # print("Simple template:")
    # print(f"  {messages}")

    # Multi-message template
    multi = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a translator. Be concise."),
            ("human", "Translate '{text}' to {language}"),
        ]
    )

    messages = multi.format_messages(text="Good morning", language="Japanese")

    print("\nMulti-message template:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content}")


demo_basic_templates()