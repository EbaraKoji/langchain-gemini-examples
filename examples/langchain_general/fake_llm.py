import subprocess

from langchain.agents import (AgentExecutor, AgentType,  # noqa F401
                              initialize_agent, load_tools)
from langchain.llms.fake import FakeListLLM
from langchain.utilities.python import PythonREPL
from langchain_core.tools import BaseTool


class DangerousPythonREPLTool(BaseTool):
    """A tool for running python code in a REPL."""
    name = "Python_REPL"
    description = ("A Python shell. Use this to execute python commands. "
                   "Input should be a valid python command. "
                   "If you want to see the output of a value, you should print it out "
                   "with `print(...)`.")

    def _run(self, command: str):
        # Use subprocess to run the Python command
        process = subprocess.Popen(["python", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Capture the output and errors
        output, errors = process.communicate()

        # Check if there were any errors
        if process.returncode != 0:
            raise Exception(f"Error executing command: {errors}")

        return output


# "python_repl" has been removed from langchain because of security vulnarabilities.
# tools = load_tools(["python_repl"])
# responses = ["Action: Python_REPL\nAction Input: print(2 + 2)", "Final Answer: 4"]

tools = [DangerousPythonREPLTool()]
answer = PythonREPL().run("print(2 + 2)")
responses = ["Action: Python_REPL\nAction Input: print(2 + 2)", f"Final Answer: {answer}"]

llm = FakeListLLM(responses=responses)

# initialize_agent() is deprecated
agent: AgentExecutor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.invoke("whats 2 + 2")
