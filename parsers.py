"""
Output Parsers and Structured Output in LangChain V.1
"""

from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def demo_str_parser():
    """Basic string output parser."""

    prompt = ChatPromptTemplate.from_template(
        "Give me a one-word answer: What color is the sky?"
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({})
    print(f"Result: '{result}' (type: {type(result).__name__})")


def demo_json_parser():
    """JSON output parser."""

    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with keys 'city' and 'country' for: {place}\n"
        "Return ONLY valid JSON, no explanation."
    )

    parser = JsonOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"place": "Tajmahal"})
    print(f"Result: {result}")
    print(f"City: {result['city']}, Country: {result['country']}")

def demo_pydantic_parser():
    """Pydantic output parser for type-safe structured data."""

    """Create a simple recipe for: {dish}"""

    # Define schema
    class Receipe(BaseModel):
        name: str = Field(description="Name of the recipe")
        ingredients: List[str] = Field(description="List of ingredients")
        prep_time_minutes: int = Field(description="Preparation time in minutes")
        difficulty: str = Field(description="easy, medium, or hard")

    parser = PydanticOutputParser(pydantic_object=Receipe)

    prompt = ChatPromptTemplate.from_template(
        "Create a simple recipe for: {dish}\n\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser

    result = chain.invoke({"dish": "scrambled eggs"})
    
    print(f"Recipe: {result.name}")
    print(f"Ingredients: {result.ingredients}")
    print(f"Prep time: {result.prep_time_minutes} mins")
    print(f"Difficulty: {result.difficulty}")

def demo_structured_output():
    """Modern with_structured_output() method."""

    class TaskExtraction(BaseModel):
        """Extracted task information."""

        task: str = Field(description="The main task to do")
        priority: str = Field(description="high, medium, or low")
        deadline: Optional[str] = Field(description="Deadline if mentioned")
        assignee: Optional[str] = Field(description="Person assigned if mentioned")

    # Bind schema to model
    structured_model = model.with_structured_output(TaskExtraction)

    # No need for format instructions - it's automatic
    prompt = ChatPromptTemplate.from_template("Extract task information from: {text}")

    chain = prompt | structured_model

    texts = [
        "John needs to finish the report by Friday - it's urgent",
        "We should update the docs sometime next week",
        "Critical: Fix the login bug ASAP",
    ]

    print("Task Extractions:")
    for text in texts:
        result = chain.invoke({"text": text})
        print(f"\nInput: {text}")
        print(f"  Task: {result.task}")
        print(f"  Priority: {result.priority}")
        print(f"  Deadline: {result.deadline}")
        print(f"  Assignee: {result.assignee}")



if __name__ == "__main__":
    print("=" * 50)
    print("Demo 1: String Parser")
    print("=" * 50)
    demo_str_parser()

    print("\n" + "=" * 50)
    print("Demo 2: JSON Parser")
    print("=" * 50)
    demo_json_parser()

    print("\n" + "=" * 50)
    print("Demo 3: Pydantic Parser")
    print("=" * 50)
    demo_pydantic_parser()

    print("\n" + "=" * 50)
    print("Demo 3: Pydantic Parser")
    print("=" * 50)
    demo_structured_output()