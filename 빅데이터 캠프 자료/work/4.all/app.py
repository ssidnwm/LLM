# 사전 셋팅
# 0.pdf에서 만든 db를 복사해옵니다.

import asyncio
from typing import AsyncIterable

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
import os

# langchain의 기록을 확인할 수 있습니다.
import langchain
langchain.debug = True

# 환경변수를 만듭니다.
os.environ["OPENAI_API_KEY"] = ""
load_dotenv()

# 임베딩 생성
embeddings = OpenAIEmbeddings()

# DB 불러오기 및 검색기 생성
vectorstore = FAISS.load_local('./db/faiss', embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# API 서버 관련 셋팅
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    content: str


async def send_message(content: str) -> AsyncIterable[str]:

    # 모델을 불러옵니다.
    callback = AsyncIteratorCallbackHandler()
    llm = ChatOllama(model="EEVE-Korean-10.8B:latest", streaming=True, verbose=True, callbacks=[callback],)

    # LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
    prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.


#Context: 
{context} 

#Question: 
{question} 

#Answer:""")


    # LangChain 표현식 언어 체인 구문을 사용합니다.
    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
    task = asyncio.create_task(
        chain.ainvoke(content)
    )


    # API에서 steram형식으로 출력되게끔 설정합니다.
    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task


# API 서버 생성
@app.post("/stream_chat/") # 경로의 이름은 stream_chat입니다. request_test.py 파일에서 stream_chat을 붙여서 request 해야합니다.
async def stream_chat(message: Message):
    generator = send_message(message.content)
    return StreamingResponse(generator, media_type="text/event-stream")


# api서버 띄우기 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)