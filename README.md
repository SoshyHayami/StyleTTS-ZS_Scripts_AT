# StyleTTS ZS: Acoustic Synth scripts



## Done and to be Done

- [x] Turned the acoustic synthesizer's notebook into scripts in the format of classic STTS repos
- [x] Saving, Logging and Loading the checkpoints so far seem to be working good
- [x] added the config to the training script
- [ ] Only DP so far, may add Accelerate later 
- [ ] pyannote's SV didn't work, replaced it with another instance of WavLM SV

## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/SoshyHayami/StyleTTS-ZS.git
cd StyleTTS2
```
3. Install python requirements: 
```bash
pip install -r requirements.txt
```

## Training
acousting synth training:
```bash
python train_.py --config_path ./Configs/config.yml
```
