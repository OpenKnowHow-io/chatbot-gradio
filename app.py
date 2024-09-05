from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# Load model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


chat_template = """
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {%- endif %}
{%- endfor %}
"""

# Create the Jinja2 template
# template = Template(chat_template)
tokenizer.chat_template = chat_template
template = tokenizer.chat_template
template = template.replace("SYS", "SYSTEM")  # Change the system token
tokenizer.chat_template = template  # Set the new template


def vanilla_chatbot(message, history):
    # Prepare the input
    chat_history = []
    for human, assistant in history:
        chat_history.append({"role": "user", "content": human})
        chat_history.append({"role": "assistant", "content": assistant})
    chat_history.append({"role": "user", "content": message})

    # Tokenize and generate
    inputs = tokenizer.apply_chat_template(chat_history, return_tensors="pt")

    outputs = model.generate(
        inputs, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2
    )

    # Decode and return the response
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response


demo_chatbot = gr.ChatInterface(
    vanilla_chatbot,
    title="Vanilla Chatbot",
    description="Enter text to start chatting.",
)

demo_chatbot.launch(debug=True, share=False)
