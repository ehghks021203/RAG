import os
import sys
import openai
import chromadb
from chromadb.utils import embedding_functions
sys.path.append(os.path.dirname(os.path.abspath('')))
from config import Chatbot, API

# ChatGPT Setting
openai.api_key = API.OPENAI_KEY
MODEL = Chatbot.MODEL

def chat(msg: str, history: list = []):
    messages = []
    SYSTEM_MSG = Chatbot.SYSTEM_PROMPT

    messages.append({"role":"system", "content":SYSTEM_MSG})
    history.append({"role":"user", "content":msg})
    messages.extend(history)

    chatbot = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages
    )

    bot_msg = chatbot["choices"][0]["message"]["content"]
    history.append({"role":"assistant", "content":bot_msg})
    
    return bot_msg, history

def text_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return response["data"][0]["embedding"]

def rag(msg: str, history: list = []):
    messages = []
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-ada-002"
            )

    client = chromadb.PersistentClient(path=Chatbot.CHROMA_DB_PATH)
    disease_collection = client.get_collection("disease",embedding_function=openai_ef)
    medicine_collection = client.get_collection("medicine",embedding_function=openai_ef)

    # 사용자 입력
    vector = text_embedding(msg)

    results = disease_collection.query(    
        query_embeddings = vector,
        n_results = 3,
        include = ["documents"]
    )

    disease_res = "\n".join(str(item) for item in results["documents"][0])
    
    results = medicine_collection.query(    
        query_embeddings = vector,
        n_results = 5,
        include = ["documents"]
    )

    medicine_res = "\n".join(str(item) for item in results["documents"][0])

    sys_prompt = f'''
    {Chatbot.SYSTEM_PROMPT}

    사용자가 증상에 대해 물어보면 너는 주어진 질병 관련 Context를 바탕으로 짧게 요약해서 중요한 정보만 답해줘야해.

    질병 관련 Context:
    {disease_res}
    
    또한, 의약품을 추천해 줄 때에는 주어진 의약품 관련 Context를 바탕으로 짧게 요약해서 중요한 정보만 답해줘야해.
    
    의약품 관련 Context:
    {medicine_res}
    '''
    with open("./prompt.txt", "w") as f:
        f.write(sys_prompt)

    messages.append({"role":"system", "content":sys_prompt})
    history.append({"role":"user", "content":msg})
    messages.extend(history)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.1
    )
    bot_msg = response["choices"][0]["message"]["content"]
    history.append({"role":"assistant", "content":bot_msg})

    return bot_msg, history

if __name__ == "__main__":
    first = True
    history = []
    while True:
        if first:
            print("Bot >> ", end="")
            msg, history = rag("안녕", history)
            print(msg)
            first = False
        print("Message >> ", end="")
        input_str = input();
        print("Bot >> ", end="")
        msg, history = rag(input_str, history)
        print(msg)