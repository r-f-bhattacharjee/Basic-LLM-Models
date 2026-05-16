from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm1=OpenAI(model='gpt-3.5-turbo-instruct')
results=llm1.invoke('What is Paytm')
print(results)

