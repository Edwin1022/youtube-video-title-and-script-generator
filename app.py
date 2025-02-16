import os

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# App framework
st.title('🦜️🔗 YouTube Video and Script Generator')
prompt = st.text_input('Plug in your video idea here')

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template="Write me a youtube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template="Write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia research: {wikipedia_research}"
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.9, max_tokens=3000)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper(max_tokens=3000)

# Show stuff to the screen if there's a prompt
if prompt:
    if len(prompt) > 1096:
        st.error("Prompt exceeds the maximum allowed length. Please shorten your prompt.")
    else:
        with st.spinner('Generating video title and script... Please wait...'):
            title = title_chain.run(topic=prompt)
            wikipedia_research = wiki.run(prompt)
            script = script_chain.run(title=title, wikipedia_research=wikipedia_research)

        st.subheader('Generated Video Title')
        st.write(title)

        st.subheader('Generated Video Script')
        st.write(script)

        with st.expander('Title History'):
            st.info(title_memory.buffer)
        
        with st.expander('Script History'):
            st.info(script_memory.buffer)