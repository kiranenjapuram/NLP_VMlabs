import os
import openai
import gradio as gr
from decouple import config

#if you have OpenAI API key as an environment variable, enable the below
openai.api_key = os.getenv("OPENAI_API_KEY")

#if you have OpenAI API key as a string, enable the below
#openai.api_key = config("OPENAI_API_KEY")
#print(openai.api_key)

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

prompt = ""

def openai_create(prompt):
    # print(prompt)

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.7,
    max_tokens=1800,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
    )
    # print(response)
    return response.choices[0].text



def chatgpt_clone(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    #output = openai_create(input) # no history in previous context.
    output = openai_create(inp) #keeps history in previous context.

    history.append((input, output))
    return history, history


block = gr.Blocks()


with block:
    gr.Markdown("""<h1><center>Build Yo'own ChatGPT with OpenAI API & Gradio</center></h1>
    """)
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder=prompt)
    state = gr.State()
    submit = gr.Button("SEND")
    submit.click(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state])

block.launch(debug = True)
