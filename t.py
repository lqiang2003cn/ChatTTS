import ChatTTS
import torch
import torchaudio

###################################
# Sample a speaker from Gaussian.
chat = ChatTTS.Chat()
chat.load(compile=False)  # Set to True for better performance

texts = ["i want to eat an apple", "book"]

wavs = chat.infer(texts)

for i in range(len(wavs)):
    torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)

rand_spk = chat.sample_random_speaker()
print(rand_spk)  # save it for later timbre recovery

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=rand_spk,  # add sampled speaker
    temperature=.3,  # using custom temperature
    top_P=0.7,  # top P decode
    top_K=20,  # top K decode
)

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7)
# to generate special token in text to synthesize.
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)

wavs = chat.infer(
    texts,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)

###################################
# For word level manual control.

text = '[uv_break][uv_break][uv_break]book[uv_break][uv_break][uv_break]'
wavs = chat.infer(text, skip_refine_text=True, params_refine_text=params_refine_text, params_infer_code=params_infer_code)
torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
