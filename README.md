# pytorch_learning

## Set up

I'm gonna assume you're cool and on Linux.
1. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for great justice.
2. Set up the Python environment
   ```shell
   python -m venv venv
   source venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   You can just copy and paste this safely.
3. To test everything is working, try `python sample_tutorial.py`. This will download a sample dataset, train a model on it, and have the model guess at the contents of an image in the dataset.
