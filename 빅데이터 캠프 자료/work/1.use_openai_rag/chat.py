# 출처 : 테디노트

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os

# embedding한 데이터 로드
os.environ["OPENAI_API_KEY"] = (
    ""
)

# 임베딩 모델 불러오기
embeddings = OpenAIEmbeddings()

# 검색기 생성
vectorstore = FAISS.load_local(
    "./db/faiss", embeddings=embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

# 프롬프트 생성
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
)

# 언어모델 생성
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 체인 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "어린왕자는 어디서 왔나요?"

# 체인을 실행합니다.
response = chain.invoke(question)
print(response)
