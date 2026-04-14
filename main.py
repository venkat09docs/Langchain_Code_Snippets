from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

def main():
    # Test openai
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatOpenAI: {response}")

    # Test anthropic
    llm_anthropic = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    response_anthropic = llm_anthropic.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatAnthropic: {response_anthropic}")


def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and Runnables."""

    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    parser = StrOutputParser()

    chain = prompt | model | parser

    # Batch - run with multiple inputs
    inputs = [
        {"question": "Hello, how are you?"},
        {"question": "What is your name?"},
        {"question": "Where is the nearest restaurant?"},
    ]

    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0]['question']} => Output: {text[1]}")


def demo_streaming():
    """Demonstrate streaming for real-time output."""

    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    # Streaming - run with streaming enabled
    print("Streaming output: ")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming


def demo_init_chat_model():
    chat_model = init_chat_model(
        model="gpt-4o-mini",
        # model_provider="openai",
        temperature=0.7,
        streaming=True,
        max_retries=3,
        max_tokens = 20
    )

    # easy to switch model providers
    if os.getenv("ANTHROPIC_API_KEY"):
        claude = init_chat_model(
            model="claude-sonnet-4-5-20250929",
            model_provider="anthropic",
            temperature=0.7,
            streaming=True,
            max_retries=3,
        )

    response = chat_model.invoke("What is the capital of France? Answer in one word.")
    print(f"Response: {response.content}")

def demo_multi_models():
    prompt = "Explain recursion in one sentence."

    models = {
            "gpt-4o-mini": init_chat_model(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=False,
            ),
            "gpt-4o": init_chat_model(
            model="gpt-4o",
            temperature=0.7,
            streaming=False,
        )         
    }

    # add anthropic model if available
    if os.getenv("ANTHROPIC_API_KEY"):
        models["claude-sonnet-4-5-20250929"] = init_chat_model(
            model="claude-sonnet-4-5-20250929",
            model_provider="anthropic",
            temperature=0.7,
            streaming=False,
        )

    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n")





if __name__ == "__main__":
    # main()
    # demo_basic_chain()
    # demo_streaming()
    # demo_init_chat_model()
    demo_multi_models()



