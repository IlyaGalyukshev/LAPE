from transformers import GPT2LMHeadModel, GPT2Tokenizer

                

tokenizer = GPT2Tokenizer.from_pretrained("ai-forever/mGPT-1.3B-tatar", use_fast=True)
model = GPT2LMHeadModel.from_pretrained("ai-forever/mGPT-1.3B-tatar").to("cuda")

text = "Александр Сергеевич Пушкин родился в "
input_ids = tokenizer.encode(text, return_tensors="pt").cuda()
out = model.generate(
        input_ids, 
        min_length=100, 
        max_length=100, 
        eos_token_id=5, 
        top_k=10,
        top_p=0.0,
        no_repeat_ngram_size=5
)
generated_text = list(map(tokenizer.decode, out))[0]
print(generated_text)