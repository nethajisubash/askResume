from openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

st.title("Subash Resume GPT")

#initialize model
if 'model' not in st.session_state:
    st.session_state['model'] = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#initialize messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

#setup sidebar
st.sidebar.title("model parameters")
temp = st.sidebar.slider("Temperature", min_value=0.0,max_value=2.0,value=0.7, step=0.1)
max_tokens = st.sidebar.slider('Max Tokens', min_value=1, max_value=4096, value=256)

#message history
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

#get input
if prompt := st.chat_input("Enter your query about Subash"):
    st.session_state['messages'].append({'role':'user','content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        client = st.session_state['model']
        stream = client.chat.completions.create(
            model = 'gpt-3.5-turbo',
            messages=[
                {"role":message["role"],"content":message["content"]} for message in st.session_state['messages']],
            temperature=temp,
            max_tokens=max_tokens,
            stream=True
        )

        response = st.write_stream(stream)
    st.session_state['messages'].append({'role': "assistant","content": response})