import torch
import time
import librosa
import numpy as np
import re
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


def test_granite_fixed():
    model_id = "ibm-granite/granite-speech-4.1-2b"
    audio_file = "data/raw/audio/Betis-Barcelona.mp3"

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype).to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    try:
        speech_array, sr = librosa.load(audio_file, sr=16000)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    chunk_len = 30 * sr
    full_transcription = []
    total_time = 0.0

    for i in range(0, len(speech_array), chunk_len):
        chunk = speech_array[i:i + chunk_len]
        if len(chunk) < sr:
            continue

        timestamp = f"[{i // sr // 60:02d}:{i // sr % 60:02d}]"

        if np.max(np.abs(chunk)) < 0.01:
            continue

        start_t = time.time()
        try:
            chat = [{"role": "user", "content": "Transcribe the following speech into English text.<|audio|>"}]
            prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

            inputs = processor(
                audio=chunk,
                text=prompt,
                return_tensors="pt"
            )

            for k, v in inputs.items():
                inputs[k] = v.to(device)
                if inputs[k].is_floating_point():
                    inputs[k] = inputs[k].to(dtype)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    repetition_penalty=1.1,
                    do_sample=False
                )

            text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            text = text.replace("Transcribe the following speech into English text.", "").strip()
            text = re.sub(r'\[.*?\]', '', text).strip()

            if text and len(text) > 5:
                full_transcription.append(f"{timestamp} {text}")

        except Exception:
            pass

        total_time += time.time() - start_t

    if full_transcription:
        print("\n".join(full_transcription))

    if total_time > 0:
        print(f"Inference complete. Total time: {total_time:.2f}s")


if __name__ == "__main__":
    test_granite_fixed()