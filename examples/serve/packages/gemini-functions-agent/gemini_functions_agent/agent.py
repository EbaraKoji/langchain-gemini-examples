from typing import List, Tuple

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from .chat_template import react_chat_gemini_template


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str,
                             str]] = Field(...,
                                           extra={"widget": {
                                               "type": "chat",
                                               "input": "input",
                                               "output": "output"
                                           }})


duckduckgo_tool = DuckDuckGoSearchResults()
tools = [duckduckgo_tool]

llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-pro")
prompt = ChatPromptTemplate.from_template(react_chat_gemini_template)
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(input_type=AgentInput)
