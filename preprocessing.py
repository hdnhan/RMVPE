import random
import typing as T
from glob import glob
from pathlib import Path
import shutil

import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from argparse import ArgumentParser

"""
data_dir/
    MIR-1K/
        Wavfile/
        PitchLabel/
    SPEECH DATA
        FEMALE
        MALE

After preprocessing,
data_dir/
    train/
        *.wav
        *.npy
    test/
        *.wav
        *.npy

train/test: 80/20
sample rate: 16kHz
"""

random.seed(42)
np.random.seed(42)


train_iter = 0
test_iter = 0
hop_size = 0.01  # 10ms hop length
sr = 16000  # sample rate

config = ArgumentParser()
config.add_argument("--data_dir", type=str, default="data")
config.add_argument("--name", type=str, default="MIR-1K")


def preprocess(args, files: T.List[str], dtype: str = "train") -> None:
    Path(f"{args.data_dir}/{dtype}").mkdir(parents=True, exist_ok=True)
    for path in tqdm(files, total=len(files)):
        audio = librosa.load(path, sr=sr)[0]

        if "MIR-1K" == args.name:
            # hop lenth is 20 ms
            pitch_path = path.replace("Wavfile", "PitchLabel").replace(".wav", ".pv")
            pitch = np.loadtxt(pitch_path)
            freq = np.where(pitch > 0, (2 ** ((pitch - 69) / 12)) * 440, 0)
        elif "SPEECH DATA" == args.name:
            # hop length is 10 ms
            pitch_path = path.replace(".wav", ".f0").replace("mic_", "ref_").replace("MIC", "REF")
            pitch = np.loadtxt(pitch_path)
            freq = pitch[:, 0]
        else:
            raise ValueError(f"Invalid dataset name: {args.name}")

        if dtype == "train":
            global train_iter
            np.save(f"{args.data_dir}/{dtype}/{train_iter}.npy", freq)
            sf.write(f"{args.data_dir}/{dtype}/{train_iter}.wav", audio, sr)
            train_iter += 1
        else:
            global test_iter
            np.save(f"{args.data_dir}/{dtype}/{test_iter}.npy", freq)
            sf.write(f"{args.data_dir}/{dtype}/{test_iter}.wav", audio, sr)
            test_iter += 1


if __name__ == "__main__":
    args = config.parse_args()

    assert args.name in ["MIR-1K", "SPEECH DATA"]

    # Remove "train" and "test" directories
    shutil.rmtree(f"{args.data_dir}/train", ignore_errors=True)
    shutil.rmtree(f"{args.data_dir}/test", ignore_errors=True)

    if args.name == "MIR-1K":
        # MIR-1K
        mir1k_dir = f"{args.data_dir}/MIR-1K"
        files = glob(f"{mir1k_dir}/Wavfile/*.wav")
        random.shuffle(files)
        train_files = files[: int(0.8 * len(files))]
        test_files = files[int(0.8 * len(files)) :]
        print(f"MIR-1K => Train: {len(train_files)}, Test: {len(test_files)}")
        preprocess(args, train_files, "train")
        preprocess(args, test_files, "test")
    elif args.name == "SPEECH DATA":
        # SPEECH DATA
        ptdb_dir = f"{args.data_dir}/SPEECH DATA"
        files = glob(f"{ptdb_dir}/*/MIC/*/*.wav")
        random.shuffle(files)
        train_files = files[: int(0.8 * len(files))]
        test_files = files[int(0.8 * len(files)) :]
        print(f"SPEECH DATA => Train: {len(train_files)}, Test: {len(test_files)}")
        preprocess(args, train_files, "train")
        preprocess(args, test_files, "test")
