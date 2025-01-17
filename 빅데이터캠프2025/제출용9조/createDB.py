import os
import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = 'your api key'

# tiktoken을 이용한 텍스트 길이 계산 함수 정의
tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# JSONL 파일 경로 지정
loader = PyPDFLoader("/content/drive/MyDrive/projectcamp/outdata.pdf")
pages = loader.load_and_split()



# 문서 분할 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=tiktoken_len
)
split_documents = text_splitter.split_documents(pages)

# OpenAI 임베딩 모델 설정 (text-embedding-3-small 모델 지정)
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma 벡터 스토어 생성 및 문서 인덱싱
vectorstore = Chroma.from_documents(
    documents=split_documents,
    embedding=openai_embeddings,
    persist_directory='/content/drive/MyDrive/projectcamp/db/chroma/outdata'
)

# 벡터 스토어를 로컬에 저장
vectorstore.persist()

print("벡터 스토어가 성공적으로 생성되어 '/content/drive/MyDrive/projectcamp/db/chroma/outdata'에 저장되었습니다.")
