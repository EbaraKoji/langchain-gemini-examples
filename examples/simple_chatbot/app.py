import streamlit as st
from langchain.chains import ConversationChain
from langchain.embeddings.vertexai import VertexAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_google_vertexai import VertexAI

llm = VertexAI(model_name='gemini-pro')
embeddings = VertexAIEmbeddings()

memory = ConversationBufferMemory(return_messages=True)

if 'conversations' not in st.session_state:
    st.session_state['conversations'] = []
else:
    for message in st.session_state['conversations']:
        memory.save_context({'input': message['human']}, {'output': message['ai']})

conversation = ConversationChain(llm=llm, verbose=True, memory=memory)

if len(conversation.memory.load_memory_variables({})['history']) == 0:
    st.chat_message('assistant').markdown('Ask anything!')

for message in conversation.memory.load_memory_variables({})['history']:
    avatar = 'user' if message.type == 'human' else 'assistant'
    st.chat_message(avatar).write(message.content)

if user_query := st.chat_input(placeholder='Send me some messages!'):
    st.chat_message('user').write(user_query)
    response = conversation.predict(input=user_query)
    st.chat_message('assistant').markdown(response)

    conversation.memory.save_context({'input': user_query}, {'output': response})
    st.session_state['conversations'].append({'human': user_query, 'ai': response})
