from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI

load_dotenv()

# to use vertexai, please login with gcloud.
# https://cloud.google.com/python/docs/reference/aiplatform/latest
# https://googleapis.dev/python/google-api-core/latest/auth.html

template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm = VertexAI(model_name="gemini-pro")
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
llm_chain.run(question)
