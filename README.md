## This repo is intednded to be my personal implentation of the 2016 CVPR paper, called Social LSTM: Human Trajectory Prediction in Crowded Spaces

the annotated paper is attached to this repo as well
 
## Papers logic
 - Pooling based LSTM umodel
 - video mostly exaplaning the logic behind LSTMs https://www.youtube.com/watch?v=YCzL96nL7j0 


## Data sets
Data sets that have been used for training
 1. ETH Dataset and the link - https://mubbasir.github.io/HTP-benchmark/downloads/
 2. UCY Dataset and the link - https://opendatalab.com/OpenDataLab/UCY

## Training

The repository includes a small generated trajectory dataset, so training can
be run before downloading ETH or UCY data as test:

```bash
pip install -r requirenments.txt
python src/main.py --epochs 5
```

Use `python src/main.py --help` to change the number of scenes, pedestrians,
sequence length, batch size, or learning rate.

in progress 
## Line by lien code Explanation

LSTM class definition but the main code is using the Torch.utils
```python
class LSTMUnit(L.LightningModule):
```