# Conv-TasNet
A PyTorch implementation of ["TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation"](https://arxiv.org/abs/1809.07454), by Yi Luo and Nima Mesgarani.

## Install
- PyTorch 0.4.1+
- Python3 (Recommend Anaconda)
- `pip install -r requirements.txt`
- If you need to convert wjs0 to wav format and generate mixture files, `cd tools; make`

## Usage
If you already have mixture wsj0 data:
1. `$ cd egs/wsj0`, modify wsj0 data path `data` to your path in the beginning of `run.sh`.
2. `$ bash run.sh`, that's all!

If you just have origin wsj0 data (sphere format):
1. `$ cd egs/wsj0`, modify three wsj0 data path to your path in the beginning of `run.sh`.
2. Convert sphere format wsj0 to wav format and generate mixture. `Stage 0` part provides an example.
3. `$ bash run.sh`, that's all!

You can change hyper-parameter by `$ bash run.sh --parameter_name parameter_value`.

### Workflow
Workflow of `egs/wsj0/run.sh`:
- Stage 0: Convert sphere format to wav format and generate mixture (optional)
- Stage 1: Generating json files including wav path and duration
- Stage 2: Training
- Stage 3: Evaluate separation performance
- Stage 4: Separate speech using Conv-TasNet

### Visualize loss
If you want to visualize your loss, you can use `visdom` to do that:
- Open a new terminal in your remote server (recommend tmux) and run `$ visdom`
- Open a new terminal and run `$ bash run.sh --visdom 1 --visdom_id "<any-string>"` or `$ train.py ... --visdom 1 --vidsdom_id "<any-string>"`
- Open your browser and type `<your-remote-server-ip>:8097`, egs, `127.0.0.1:8097`
- In visdom website, chose `<any-string>` in `Environment` to see your loss

## NOTE
This is still a work in progress and any contribution is welcome (dev branch is main development branch).
