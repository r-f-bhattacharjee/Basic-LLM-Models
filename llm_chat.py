from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm2=ChatOpenAI(model='gpt-4o', temperature=0.9, max_tokens=1000)
results=llm2.invoke('What is Paytm')
print(results.content)
