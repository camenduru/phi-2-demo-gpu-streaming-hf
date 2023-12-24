import spaces
from detoxify import Detoxify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import gradio as gr
from threading import Thread
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.backends.mps.is_available():
    device = "mps"

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif'],
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
).to(device)
@spaces.GPU(enable_queue=True)
def generate_text(text, temperature, maxLen):
    mdl = Detoxify('original', device='cuda')
    if mdl.predict(text)['toxicity'] > 0.7:
        raise gr.Error("Sorry, our systems may have detected toxic content. Please try a different input.")
    inputs = tokenizer([text], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=maxLen, temperature=temperature)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    t = ""
    toks = 0
    for out in streamer:
        toks += 1
        if toks == 3:
            toks = 0
            if mdl.predict(t)['toxicity'] > 0.7:
                raise gr.Error("Sorry, our systems may have detected toxic content. Please try a different input.")
        t += out
        yield t
with gr.Blocks(css="footer{display:none !important}", theme=theme) as demo:
    gr.Markdown("""
# (Unofficial) Demo of Microsoft's Phi-2 on GPU

Not affiliated with Microsoft!

This model is licensed under the [Microsoft Research License](https://huggingface.co/microsoft/phi-2/resolve/main/LICENSE). You may only use it for non-commercial purposes.

Note: for longer generation (>512), keep clicking "Generate!" The demo is currently limited to 512 demos per generation to ensure all users have access to this service. Please note that once you start generating, you cannot stop generating until the generation is done.

By [mrfakename](https://twitter.com/realmrfakename). Inspired by [@randomblock1's demo](https://huggingface.co/spaces/randomblock1/phi-2).

Duplicate this Space to skip the wait!
""".strip())
    gr.DuplicateButton()
    text = gr.Textbox(label="Prompt", lines=10, interactive=True, placeholder="Write a detailed analogy between mathematics and a lighthouse.")
    temp = gr.Slider(label="Temperature", minimum=0.1, maximum=1.5, value=0.7)
    maxlen = gr.Slider(label="Max Length", minimum=4, maximum=512, value=75)
    go = gr.Button("Generate", variant="primary")
    go.click(generate_text, inputs=[text, temp, maxlen], outputs=[text], concurrency_limit=2)
    examples = gr.Examples(
        [
            ['Write a detailed analogy between mathematics and a lighthouse.', 0.7, 75],
            ['Instruct: Write a detailed analogy between mathematics and a lighthouse.\nOutput:', 0.7, 75],
            ['Alice: I don\'t know why, I\'m struggling to maintain focus while studying. Any suggestions?\n\nBob: ', 0.6, 150],
            ['''def print_prime(n):
   """
   Print all primes between 1 and n
   """\n''', 0.2, 100],
        ],
        [text, temp, maxlen]
    )

if __name__ == "__main__":
    demo.queue().launch(show_api=False)

