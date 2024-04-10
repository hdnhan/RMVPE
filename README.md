# RMVPE
Download `MIR-1K.zip` and `SPEECH_DATA_ZIPPED.zip` then extract them to `data/` folder.

- For hop size 320 (20ms):
```bash
git checkout lstm
python preprocess.py --name "MIR-1K"
python train.py
```

- For hop size 160 (10ms):
```bash
git checkout lstm-10
python preprocess.py --name "SPEECH DATA"
python train.py
```
