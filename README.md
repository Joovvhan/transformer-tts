# Transformer-TTS (and more!)

#### We plans to implement some TTS Deep Learning algorithms here
* Transformer-TTS
* FastSpeech
* FastSpeech2
* ...

#### TODO

- [x] Build Dataset Loader
- [x] Compare mel-spectrogram processing/loading time (3:1)
- [ ] Build a model and modules
- [ ] Baseline model architecture
- [ ] Tensorboard logging
- [ ] requirements.txt or Docker image
- [ ] overwrite configs with parsed arguments
- [ ] Check why phoneme dictionary is of length 12463
- [ ] Make phoneme dictionary process multi-threaded

#### SETUP
1. git clone https://github.com/Joovvhan/transformer-tts.git
2. cd transformer-tts
3. source scripts/set_locale.sh
4. source scripts/init.sh 
5. python main.py

#### Reference
* Neural Speech Synthesis with Transformer Network
* Each phoneme has a trainable embedding of 512 dims
* the output of each convolution layer has 512 channels, followed by a batch normalization and ReLU activation, and a dropout layer as well.
* we add a linear projection after the final ReLU activation
