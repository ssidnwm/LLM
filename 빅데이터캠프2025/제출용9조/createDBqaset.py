import os
import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = 'your api key'

# tiktoken을 이용한 텍스트 길이 계산 함수 정의
tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# JSONL 파일 경로 지정
file_path = '/content/drive/MyDrive/projectcamp/kosaf_faq.txt'

# 텍스트 파일에서 데이터 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    text_data = file.read()

# 텍스트를 Document 객체로 변환
documents = [Document(page_content=text_data, metadata={"source": file_path})]
# 문서 분할 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=tiktoken_len
)
split_documents = text_splitter.split_documents(documents)

# OpenAI 임베딩 모델 설정 (text-embedding-3-small 모델 지정)
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma 벡터 스토어 생성 및 문서 인덱싱
vectorstore = Chroma.from_documents(
    documents=split_documents,
    embedding=openai_embeddings,
    persist_directory='/content/drive/MyDrive/projectcamp/db/chroma/qaset'
)

# 벡터 스토어를 로컬에 저장
vectorstore.persist()

print("벡터 스토어가 성공적으로 생성되어 '/content/drive/MyDrive/projectcamp/db/chroma/qaset'에 저장되었습니다.")