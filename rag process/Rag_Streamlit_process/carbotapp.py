import os
import json
import streamlit as st
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import StreamlitChatMessageHistory
from loguru import logger
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
import functools
# Constants and file paths
Hyundai_logo = "images/Hyundai_logo.png"
horizontal_logo = "images/Hyundai_logo_horizen.png"
filepath_pChunk_content = "/home/r22000245/fullDocument_pChunk.json"
filepath_pChunk_IDs = "/home/r22000245/fullDocument_mapping.json"

# Load the vector store
def load_vectorstore(db_path):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(persist_directory=db_path, embedding_function=embeddings)

# Load vector store into session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore('db/chroma/cchunk')

# Set up retriever tool
retriever = st.session_state.vectorstore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_chunk_data",
    "Retrieve relevant chunks from the vector store based on user queries on UNECE legislation for braking devices."
)

# Define helper function to format tool for OpenAI functions
def format_tool_to_openai_function(tool):
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema.schema(),
    }

# Use retriever_tool directly as tool in tools list for ToolNode
tools = [retriever_tool]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Other helper functions for retrieving and formatting chunks
def retrieve_cChunk_IDs(query):
    st.write(f"Retrieving cChunks for query: {query}")
    search_results = retriever.similarity_search(query, k=4)
    if not search_results:
        st.write("No cChunks retrieved for the given query.")
    else:
        st.write(f"Retrieved cChunks: {[result.page_content for result in search_results]}")
    IDs_matched = [json.loads(result.page_content)["ID"] for result in search_results]
    return IDs_matched

def get_pChunk(pChunk_dbPath, pChunk_IDsPath, cChunk_IDs):
    st.write(f"Loading pChunks for cChunk IDs: {cChunk_IDs}")
    with open(pChunk_dbPath, 'r') as infile:
        parent_chunks = json.load(infile)
    with open(pChunk_IDsPath, 'r') as id_file:
        child_to_parent_map = json.load(id_file)

    set_target_pChunk = ""
    for cChunk_ID in cChunk_IDs:
        parent_chunk_id = next((parent for parent, children in child_to_parent_map.items() if cChunk_ID in children), None)
        if parent_chunk_id and parent_chunk_id in parent_chunks:
            set_target_pChunk += str(parent_chunks[parent_chunk_id]) + "\n\n"
    if not set_target_pChunk:
        st.write("No pChunks matched for the given cChunk IDs.")
    return set_target_pChunk

# Grade documents based on relevance
def grade_documents(state, openai_api_key) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    st.write("Grading documents for relevance.")
    st.write("grade input state:", st.session_state['state'])
    class GradeModel(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    model = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18", openai_api_key=openai_api_key)
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. 
        Here is the retrieved document: {context} 
        Here is the user question: {question} 
 Your task:
        1. Answer "yes" if the document contains enough information to directly or indirectly answer the question, even if the answer to the question itself is "no".
        2. Answer "no" if the document does not contain sufficient information to answer the question, either directly or indirectly.

        Example 1:
        Question: "Is the capital of Spain Madrid?"
        Document: "Spain is a country in Europe. Its geography includes mountains and rivers."
        Grading: "no" (Insufficient information to answer the question)

        Example 2:
        Question: "Is 11% of failed test runs in vehicle-to-vehicle testing acceptable?"
        Document: "The number of failed test runs for Car to Car tests must not exceed 10%."
        Grading: "yes" (Document provides enough information to conclude that 11% is not acceptable)

        Example 3:
        Question: "Is the capital of France Paris?"
        Document: "France's capital city is Paris."
        Grading: "yes" (Document directly answers the question)

        Now, based on the above rules, grade the document as 'yes' or 'no'. Make sure to only create answers as yes or no. No other additional explanation is required.
        """,
        input_variables=["context", "question"]
    )
    
    messages = state["messages"]
    if isinstance(messages[0], HumanMessage):
        question = messages[0].content  # 올바르게 유저 질문을 가져옴
    else:
        # 메시지가 잘못된 위치에 있을 경우 사용자 메시지 찾기
        question = next((msg.content for msg in messages if isinstance(msg, HumanMessage)), None)
        if not question:
            st.write("Error: Unable to find user question.")
            return "rewrite"




    docs = messages[-1].content  # 마지막 검색된 문서 가져오기
    #st.write(f"Question: {question}")
    #st.write(f"Documents to grade: {docs}")
    # Invoke chain and parse response manually
    response = prompt | model
    result = response.invoke({"question": question, "context": docs}).content
    st.write(f"Grading response: {result}")
    st.write(f"After grade state:", state) 
    try:
        parsed_response = GradeModel(binary_score=result.strip())
    except ValueError:
        return "rewrite"  # Parsing 실패 시 rewrite로 이동
    try:
        # 처음에는 원래의 result를 파싱 시도
        parsed_response = GradeModel(binary_score=result.strip())  
        cleaned_score = parsed_response.binary_score.strip('"').strip().lower()
        if cleaned_score == "yes":
            st.write("Grading result indicates 'generate'.")
            return "generate"
        else:
            st.write("Grading result indicates 'rewrite'.")
            st.write("Debugging Messages in State:")
            #for idx, msg in enumerate(state["messages"]):
            #    st.write(f"Message {idx}: Type: {type(msg)}, Content: {getattr(msg, 'content', 'No content')}")
            return "rewrite"

    except ValueError:
        # ValueError가 발생한 경우 로그를 통해 확인
        st.write("Initial parsing failed due to ValueError. Attempting to remove prefix.")

        # 접두사 'Grading: ' 제거하고 다시 파싱 시도
        clean_result = result.replace('Grading: ', '').strip()

        try:
            parsed_response = GradeModel(binary_score=clean_result)
            st.write(f"Parsed response after cleaning: {parsed_response.binary_score}")
            if parsed_response.binary_score.lower() == "yes":
                st.write("Cleaned grading result indicates 'generate'.")
                return "generate"
            else:
                st.write("Cleaned grading result indicates 'rewrite'.")
                return "rewrite"

        except ValueError:
            # 여전히 파싱 실패 시, rewrite로 이동
            st.write("Parsing failed again even after cleaning. Proceeding with rewrite.")
            return "rewrite"

# Agent function with improved initial message handling
def agent(state, openai_api_key):
    # 사용자 질문을 처리하기 위한 초기 메시지 추가
    if not state["messages"] or not isinstance(state["messages"][-1], HumanMessage):
        return state  # 유효하지 않은 상태라면 아무것도 수행하지 않음
    
    # 모델 설정 및 함수 바인딩
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini-2024-07-18", openai_api_key=openai_api_key)
    functions = [format_tool_to_openai_function(retriever_tool)]
    llm_with_functions = model.bind_functions(functions=functions)
    
    # 모델 호출 후 응답을 AIMessage 형식으로 변환
    result = llm_with_functions.invoke(state["messages"])
    
    if result.tool_calls:
        wrapped_result = AIMessage(**result.dict(exclude={"type", "name"}), name="agent_response")
    else:
        wrapped_result = AIMessage(**result.dict(exclude={"type", "name"}), name="agent_response")
    
    # 상태 메시지에 추가하고 반환
    state["messages"].append(wrapped_result)
    return {"messages": state["messages"]}

# Workflow setup for retrieving, rewriting, and generating answers
def define_agent_workflow(openai_api_key):
    # 세션 상태가 없다면 초기화
    if 'state' not in st.session_state:
        st.session_state['state'] = {"messages": [], "used_chunk_ids": [], "final_results": []}

    # 세션 상태를 가져옵니다.
    state = st.session_state['state']

    # Define the workflow
    workflow = StateGraph(AgentState)

    # Define the nodes
    retrieve = functools.partial(retrieve_and_add_to_state, openai_api_key=openai_api_key)
    rewrite_node = functools.partial(rewrite, openai_api_key=openai_api_key)
    generate_node = functools.partial(generate, openai_api_key=openai_api_key)
    workflow.add_node("retrieve", retrieve)  # 검색된 결과가 있으면 grade_documents로 연결
    workflow.add_node("rewrite", rewrite_node)  # 질문을 다시 작성
    workflow.add_node("generate", generate_node)  # 응답 생성

    # Start by calling agent, but force it to always move to retrieve
    workflow.add_edge(START, "retrieve")  # 검색을 강제

    # Define retrieval followed by document grading and combination
    workflow.add_conditional_edges(
        "retrieve",
        lambda state: grade_documents(state, openai_api_key),  # API 키를 전달하도록 수정
        {
            "generate": "generate",  # 유효성을 통과하면 generate로 이동
            "rewrite": "rewrite",  # 유효성 평가 실패 시 rewrite로 이동
        },
    )



    # 워크플로우 실행 후 최종 상태를 업데이트합니다.
    st.session_state['state'] = state

    workflow.add_edge("generate", END)  # generate에서 종료
    workflow.add_edge("rewrite", "retrieve")  # rewrite 후 다시 검색 수행
    return workflow.compile()
# Retrieve and add to state function to fix retrieval issues
def retrieve_and_add_to_state(state,openai_api_key):
    max_retries = 3  # 최대 재시도 횟수 
    min_chunk_count = 4  # 최소 확보해야 할 청크 수
    attempt = 0
    used_chunk_ids = set(st.session_state['state'].get("used_chunk_ids", []))
    #final_results = st.session_state['state'].get("final_results", [])
    final_results = []
    st.write(f"start retrieve_and_add_to_state state:", st.session_state['state'])
    while attempt < max_retries:

        current_k = 4 + (attempt * 3)
        # rewritten_question이 있는지 확인 후 query를 설정
        query = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), state["messages"][0].content)
        st.write(f"Retrieve initiated with query (attempt {attempt + 1}): {query}")

        # 검색 수행
        #search_results = retriever.get_relevant_documents(query, k=current_k)
        vector_db = st.session_state.vectorstore  # 직접 벡터 데이터베이스 객체 사용
        search_results = vector_db.similarity_search(query, k=current_k)
        st.write(f"Number of search results retrieved with k={current_k}: {len(search_results)}")
        st.write(f"Retrieved search results: {[result.page_content for result in search_results]}")
        new_results = []
        new_chunk_ids = set()  # 새롭게 발견된 청크 ID를 저장

        # 중복 청크 필터링
        for result in search_results:
            try:
                chunk_content = json.loads(result.page_content)
                chunk_id = chunk_content.get("ID")
                if chunk_id and chunk_id not in used_chunk_ids:
                    new_results.append(result)
                    new_chunk_ids.add(chunk_id)  # 새로 추가된 청크 ID를 사용된 목록에 추가
                    #st.write(f"new find ids: {new_chunk_ids}")
            except json.JSONDecodeError:
                st.write("Error decoding JSON chunk content.")

        final_results.extend(new_results)
        # 충분한 청크가 확보되었는지 확인
        if len(new_results) >= min_chunk_count:
            st.write(f"Final filtered retrieved results (total): {[result.page_content for result in final_results]}")

            break  # 충분한 청크를 찾았으므로 반복을 종료합니다.  
 
        
        # 필터링 후의 검색 결과가 충분한지 확인
        if len(final_results) >= min_chunk_count:
            st.write(f"Final filtered retrieved results (total): {[result.page_content for result in final_results]}")

            break  # 충분한 청크를 찾았으므로 반복을 종료합니다.

        # 만약 청크가 충분하지 않으면 재시도 카운트 증가
        st.write(f"Insufficient new cChunks retrieved (found {len(new_results)}). Retrying...")
        attempt += 1

        # 질문 리라이팅 후 재시도
        if attempt < max_retries:
            st.write(f"Rewriting the question to improve results (attempt {attempt + 1}).")
            state = rewrite(state, openai_api_key)  # 질문을 리라이팅하고 상태 업데이트
    used_chunk_ids.update(new_chunk_ids)
    # 최종적으로 확보된 검색 결과를 상태에 추가
    if final_results:
        combined_content = "\n\n".join([result.page_content for result in final_results])
        retrieved_message = AIMessage(content=combined_content)
        state["messages"].append(retrieved_message)
        st.write(f"State updated with retrieved documents.")

    # 사용된 청크 ID 업데이트
        # 중복 청크 추가 후 상태 업데이트
    st.session_state['state']['used_chunk_ids'] = list(used_chunk_ids)
    st.session_state['state']['final_results'] = final_results
    # cChunk ID 추출
    cChunk_IDs = []
    id_pattern = r'"ID":\s*"([^"]+)"'  # JSON 형식일 때 ID 추출을 시도
    fallback_pattern = r'D_\w+'  # 대체 패턴 (예: "D_"로 시작하는 모든 문자열)

    for result in new_results:
        try:
            chunk_content = json.loads(result.page_content)  # JSON 형식으로 파싱 시도
            if "ID" in chunk_content:
                cChunk_IDs.append(chunk_content["ID"])
        except json.JSONDecodeError:
            # JSON 형식이 아닐 경우 정규 표현식을 사용하여 ID 추출
            match = re.search(id_pattern, result.page_content)
            if match:
                cChunk_IDs.append(match.group(1))
            else:
                # 대체 패턴으로 추출 (비JSON 텍스트에서 패턴 매칭)
                fallback_match = re.findall(fallback_pattern, result.page_content)
                if fallback_match:
                    cChunk_IDs.extend(fallback_match)
    # 상태에 cChunk_IDs 저장
    st.session_state['state']["cChunk_IDs"] = cChunk_IDs
    st.write(f"Extracted cChunk IDs: {cChunk_IDs}")
    # pChunk 결합을 수행
    if cChunk_IDs:
        state["cChunk_IDs"] = cChunk_IDs  # 상태에 cChunk_IDs 저장
        st.write(f"Extracted cChunk IDs: {cChunk_IDs}")
        p_chunk_state = combine_chunks(state)

        # 결합된 pChunk 메시지를 상태에 추가
        combined_p_chunk_message = p_chunk_state["messages"][-1]  # 마지막으로 추가된 메시지 (pChunk)
        state["messages"].append(combined_p_chunk_message)
        st.write(f"State updated with combined pChunk.")
    else:
        st.write("No documents retrieved for the given query.")
    st.write("retrieve output:", st.session_state['state']) 
    return st.session_state['state']
# Combine cChunk and pChunk function
def combine_chunks(state):
    query = state["messages"][0].content
    cChunk_IDs = state.get("cChunk_IDs", [])
    if not cChunk_IDs:
        st.write("No cChunk IDs available for combining pChunks.")
        return state
    combined_content = get_pChunk(filepath_pChunk_content, filepath_pChunk_IDs, cChunk_IDs)
    
    # Combine cChunk and pChunk results
    if combined_content:
        combined_message = AIMessage(content=combined_content)
        state["messages"].append(combined_message)
        st.write(f"Combined pChunk content added to state: ")
    else:
        st.write("No content to combine from cChunk and pChunk.")
    return state

# Generate function
def generate(state, openai_api_key):
    question = state["messages"][0].content
    docs = state["messages"][-1].content  # Retrieve the relevant documents
    st.write(f"Generating answer for question: {question}")
    st.write(f"Using documents:")
    # Use a language model to answer based on documents
    model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o-mini-2024-07-18", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Based on the provided context, answer the following question:\n\n
        Answer questions about automotive regulations in the following format 
        1. a summary of the question an overview of the user's request a summary of the conclusions drawn from the UN ECE documents 
        2. regulatory requirements a description of the documents on which the user's question is based 
        3. conclusions and recommendations a concise summary of the conclusions of your interpretation of the question indicating the need for additional double-checking of your interpretation
        
        
        Context: {context}\n\nQuestion: {question}"""
    )
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.run({"context": docs, "question": question})
        # 상태에 생성된 응답 추가
    if response:
        ai_response_message = AIMessage(content=response)
        state["messages"].append(ai_response_message)
    st.write(f"Generated response: {response}")
    return {"messages": [{"role": "assistant", "content": response}]}

# Rewrite function
def rewrite(state, openai_api_key):

    st.write("before rewrite", st.session_state['state'])
    original_question = state["messages"][0].content
    current_question = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        original_question
    )
    # Use a language model to rewrite the question
    model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o-mini-2024-07-18", temperature=0)
    rewrite_prompt = PromptTemplate(
        input_variables=["original_question", "current_question"],
        template="""Rewrite the following question for better clarity. 
        Make the most of the questions you're given, but be sure to make them clearer and more specific for your answers.:\n\n
        
        Original Question: {original_question}
        Current Question: {current_question}

        Rewrite it for better clarity, without losing the essence of the original question.

        Rewrite Instructions:
        1. Rewrite the question to make it significantly different in structure, while still preserving the core idea.
        2. Avoid repeating the phrasing of the original question.
        3. Introduce alternative phrasings, perspectives, or a slightly expanded context to broaden the search scope.
        4. Ensure that the rewritten question could potentially guide the model to consider different aspects of the context.

        Output Rules:
        - Maintain the essence of the question, but reword it creatively to explore related ideas or interpretations.
        - Try including different examples, focusing on overlooked details, or shifting the context slightly to get more diverse retrieval results.
        
        """
    )
    chain = LLMChain(llm=model, prompt=rewrite_prompt)
    rewritten_question = chain.run({"original_question": original_question, 
        "current_question": current_question})
    st.write(f"rewritten_questions:{rewritten_question}")
    rewritten_message = HumanMessage(content=rewritten_question)
    state["messages"].append(rewritten_message)

    st.session_state['state']["messages"] = state["messages"]
     # 기존 필드가 있다면 그대로 유지
    if "used_chunk_ids" in state:
        st.session_state['state']["used_chunk_ids"] = state["used_chunk_ids"]

    if "final_results" in state:
        st.session_state['state']["final_results"] = state["final_results"]

    if "cChunk_IDs" in state:
        st.session_state['state']["cChunk_IDs"] = state["cChunk_IDs"]
    # 상태 업데이트 후 세션 상태에 반영

    st.write(f"After rewrite state:", st.session_state['state'])
    
    return st.session_state['state']

# Running the workflow with user query
def run_agent_workflow(query, openai_api_key):
    # 초기 상태 메시지를 설정합니다.
    st.session_state['state'] = {
        "messages": [
            HumanMessage(content=query),
            AIMessage(content="User has asked a question, initiating retrieval process.")
        ],
        "used_chunk_ids": set(),  # 새로운 질문에 대해 사용된 청크 ID 초기화
        "final_results": []       # 최종 검색 결과 초기화
    }
    
    workflow = define_agent_workflow(openai_api_key)
    graph = workflow  # 컴파일된 graph 사용
    
    for _ in graph.stream(st.session_state['state']):
        # 각 노드가 `st.session_state`를 직접 수정하므로 여기서는 병합이 필요 없음
        st.write("Complete output after streaming a node:", st.session_state['state'])

        # 상태 병합 디버깅을 위한 추가 코드
        # 현재 output에서 상태 병합이 제대로 이루어지고 있는지 확인
#        for node_name, result in state.items():
#            st.write(f"Node '{node_name}' output:", result)  # 각 노드의 결과 확인

            # 결과의 키를 모두 출력해서 `final_results`가 있는지 확인
#            st.write(f"Keys in node '{node_name}' result:", list(result.keys()))

            # 상태 병합을 수행하고 그 이후 상태를 확인
#            st.write(f"Before merging state for node '{node_name}':", state)

            # 상태 병합 작업
#            for key, value in result.items():
#                if key in state:
#                    if isinstance(state[key], set) and isinstance(value, set):
#                        state[key].update(value)
#                    elif isinstance(state[key], list) and isinstance(value, list):
#                        state[key].extend(value)
#                    elif isinstance(state[key], dict) and isinstance(value, dict):
#                        state[key].update(value)
#                    else:
#                        # 기존 값이 아닌 경우, 새 값으로 상태 업데이트
#                        state[key] = value
#                else:
#                    # 기존 상태에 없는 키는 새로운 값으로 추가
#                    state[key] = value

                # 상태 병합 후 상태를 업데이트할 때의 출력
#                st.write(f"Key '{key}' updated. Current state of key '{key}':", state[key])

            # 상태 병합 후 최종 상태 확인
#            st.write(f"After merging state for node '{node_name}':", state)

        # 전체 상태를 확인하여 'final_results'가 제대로 유지되고 있는지 확인
#        st.write("Final state after all nodes processed:", state)
 
# Define a conversation chain to manage interaction in Streamlit session
def get_conversation_chain(openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4o-mini-2024-07-18', temperature=0)
    conversation_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["infos", "query"],
            template="""
                Based on the provided context, answer the following question:
                Context: {infos}
                Question: {query}
            """
        ),
        verbose=True,
    )
    return conversation_chain

# Main function to run Streamlit app
def main():
    st.set_page_config(page_title="Hyundai Motor Company - Motor Vehicle Law", page_icon=Hyundai_logo)
    st.title("Hyundai Motor - Motor Vehicle Law Data QA Chatbot")
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vectorstore('db/chroma/cchunk')
    
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
    
    if st.sidebar.button("Start Chatting") and "openai_api_key" in st.session_state:
        st.session_state.conversation = get_conversation_chain(st.session_state.openai_api_key)
    
    query = st.chat_input("질문을 입력하세요")
    
    if query and "openai_api_key" in st.session_state:
        run_agent_workflow(query, st.session_state.openai_api_key)

if __name__ == '__main__':
    main() 