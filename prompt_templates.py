from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
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

    messages = multi.format_messages(text="Good morning", language="Hindi")

    print("\nMulti-message template:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content}")

    model = init_chat_model(model="gpt-4o-mini", temperature=0)
    response = model.invoke(messages)
    print(response.content)


def demo_fewshot_prompt_template():
    """ Fewshot Prompt """

    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    fewshot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give the opposite of each word."),
            fewshot_prompt,
            ("human", "{input}"),
        ]
    )

    model = init_chat_model(model="gpt-4o-mini", temperature=0)
    response = model.invoke(final_prompt.format_messages(input="tall"))

    print(response.content)


def demo_prompt_composition():
    """Compose prompts from reusable parts."""

    # Reusable system prompt
    persona = ChatPromptTemplate.from_messages(
        [("system", "You are a {role}. Your tone is {tone}.")]
    )

    # Reusable task prompt
    task = ChatPromptTemplate.from_messages([("human", "{task}")])

    # Combine
    full_prompt = persona + task

    # Test different combinations
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    chain = full_prompt | model

    # As a pirate
    response = chain.invoke(
        {
            "role": "pirate captain",
            "tone": "adventurous",
            "task": "Tell me about your ship",
        }
    )

    print(f"Pirate: {response.content[:100]}...")

    # As a scientist
    response = chain.invoke(
        {
            "role": "HR",
            "tone": "professional",
            "task": "What is ATS based Resume?",
        }
    )
    print(f"\HR: {response.content[:100]}...")

def demo_messages_placeholder():
    """Use MessagesPlaceholder for dynamic conversation history."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Simulate conversation history
    history = [
        HumanMessage(content="My name is Venkat"),
        AIMessage(content="Nice to meet you, Venkat!"),
    ]

    messages = prompt.format_messages(history=history, question="What's my name?")

    print("With history placeholder:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content[:50]}...")
    
    # Execute
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    chain = prompt | model

    response = chain.invoke({"history": history, "question": "What's my name?"})
    print(f"\nResponse: {response.content}")


# demo_basic_templates()

# demo_fewshot_prompt_template()

# demo_prompt_composition()

demo_messages_placeholder()
