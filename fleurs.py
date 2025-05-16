import os
from pathlib import Path

import torch
from datasets import load_dataset
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from tqdm import tqdm

SPLIT = "test"
DATA_ENV_VAR = os.getenv("DATA")
DATA_ROOT = Path(DATA_ENV_VAR) / "fleurs" / "ca_es" / SPLIT
DATA_ROOT.mkdir(exist_ok=True, parents=True)

fleurs = load_dataset("google/fleurs", "ca_es", split=SPLIT, streaming=True)

device = torch.device("cuda:0")

s2vec_model_cat = SpeechToEmbeddingModelPipeline(
    encoder="sonar_speech_encoder_cat", device=device
)
s2vec_model_spa = SpeechToEmbeddingModelPipeline(
    encoder="sonar_speech_encoder_spa", device=device
)

t2vec_model = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=device,
)

for batch in tqdm(fleurs.batch(batch_size=32)):
    text_emb = t2vec_model.predict(batch["transcription"], source_lang="cat_Latn").cpu()
    arr = [torch.tensor(item["array"]) for item in batch["audio"]]
    speech_emb_cat = s2vec_model_cat.predict(
        [torch.tensor(item["array"]).unsqueeze(0) for item in batch["audio"]]
    ).cpu()
    speech_emb_spa = s2vec_model_spa.predict(
        [torch.tensor(item["array"]).unsqueeze(0) for item in batch["audio"]]
    ).cpu()

    utt_ids = batch["id"]
    wav_ids = [item["path"].split("/")[1].split(".")[0] for item in batch["audio"]]

    save = DATA_ROOT / "text_sonar_basic_encoder"
    save.mkdir(exist_ok=True)
    for utt_id, wav_id, temb in zip(utt_ids, wav_ids, text_emb):
        torch.save(temb, save / f"{utt_id}_{wav_id}.pt")

    save = DATA_ROOT / "sonar_speech_encoder_spa"
    save.mkdir(exist_ok=True)
    for utt_id, wav_id, emb in zip(utt_ids, wav_ids, speech_emb_spa):
        torch.save(emb, save / f"{utt_id}_{wav_id}.pt")

    save = DATA_ROOT / "sonar_speech_encoder_cat"
    save.mkdir(exist_ok=True)
    for utt_id, wav_id, emb in zip(utt_ids, wav_ids, speech_emb_cat):
        torch.save(emb, save / f"{utt_id}_{wav_id}.pt")
