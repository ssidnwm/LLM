from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
llm = ChatOllama(model="EEVE-Korean-10.8B:latest")

# 프롬프트 설정
prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 

#Answer:""")

# LangChain 표현식 언어 체인 구문을 사용합니다.
chain = prompt | llm | StrOutputParser()