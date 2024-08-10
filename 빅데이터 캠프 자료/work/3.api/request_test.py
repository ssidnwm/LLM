from langserve import RemoteRunnable


# langserve식 stream api request입니다.
chain = RemoteRunnable("http://74.226.172.132:8000/prompt/")

for token in chain.stream({"question":"어린왕자의 집에 대해 설명해주세요"}):
    print(token)