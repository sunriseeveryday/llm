import torch


def do_inference(tokenizer, model, prompt, max_length: int=4096):
    tokens = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        res = model.generate(
            **tokens,
            max_new_tokens=max_length - tokens.input_ids.shape[-1],
            do_sample=True,
            num_return_sequences=1,
            temperature=0.7,
            num_beams=3,
            top_p=0.95,
            early_stopping=True
        )
    return tokenizer.batch_decode(res, skip_special_tokens=True)
