import os
import csv
import librosa
import shutil
import numpy as np
from scipy.io import wavfile
from text import _clean_text
from tqdm import tqdm

def read_metadata_csv(metadata_csv_path):
    with open(metadata_csv_path, 'r', encoding="utf-8") as file:
        reader = csv.reader(file, quoting=csv.QUOTE_ALL)
        return list(reader)
    
def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    speakers = ["peterl"] # ["peterl", "peterk"]
    for speaker in speakers: 
        metadata_csv_path = os.path.join(in_dir, speaker, "metadata.csv") 
        for row in tqdm(read_metadata_csv(metadata_csv_path)):
            base_name = row[0]
            text = row[1]
            #parts = row[2]
            
            wav_path = os.path.join(in_dir, speaker, "wavs", "{}.wav".format(base_name))
            tg_path = os.path.join(in_dir, speaker, "TextGrid", "{}.TextGrid".format(base_name))
            if os.path.exists(wav_path) and os.path.exists(tg_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )

                tg_target_path = os.path.join(out_dir, speaker, f"{base_name}.TextGrid")
                shutil.copy(tg_path, tg_target_path)

                text = _clean_text(text, cleaners)
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)
                
                # with open(
                #     os.path.join(out_dir, speaker, "{}.parts".format(base_name)),
                #     "w",
                # ) as f1:
                #     f1.write(parts)