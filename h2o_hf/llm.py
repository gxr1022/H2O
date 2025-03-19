import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#/data/home/gexr/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
model_name = "/data/home/gexr/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # 避免 padding 问题

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def generate_text(prompt, max_new_tokens=1024):
    # inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    # input_ids = inputs["input_ids"].to(model.device)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "In the Kodia System in the Nashoba Sector of the Kagami Galaxy, orbiting Kodia Prime and skirting the edge, just inside, then just outside, of that main-sequence, old population I star's snow line, Kodia III is a gas giant bigger than Jupiter. Its densely forested moon is habitable due to reflected light from the gas giant and the tidal heating of the gas giant's intense gravity causing it to be a geothermal hot springs wonderland, and its atmosphere is protected by the gas giants intense magnetic field."}
    ]

    chat_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    print("Chat Prompt:", chat_prompt)
    print("Type:", type(chat_prompt))

    input_ids = torch.tensor([chat_prompt]).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,  
            max_new_tokens=max_new_tokens,  
            eos_token_id=tokenizer.eos_token_id,  
            pad_token_id=tokenizer.pad_token_id,  
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

prompt = "In the Kodia System in the Nashoba Sector of the Kagami Galaxy, orbiting Kodia Prime and skirting the edge, just inside, then just outside, of that main-sequence, old population I star's snow line, Kodia III is a gas giant bigger than Jupiter. Its densely forested moon is habitable due to reflected light from the gas giant and the tidal heating of the gas giant's intense gravity causing it to be a geothermal hot springs wonderland, and its atmosphere is protected by the gas giants intense magnetic field."
generated_text = generate_text(prompt)

print("\nGenerated Response:\n", generated_text)

