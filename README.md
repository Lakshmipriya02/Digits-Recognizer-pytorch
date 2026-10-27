
# ✍️ Handwritten Digit Recognizer

Draw a digit in the browser and a PyTorch neural network (784 → 128 → 64 → 10) predicts it live.
Trained on MNIST.

## Run locally

```bash
pip install -r requirements.txt
python train.py      # ~1-2 min on CPU, creates digits_classifier.pt
python app.py        # opens http://127.0.0.1:7860
```

## Files
- `model.py`       network definition (same as the notebook)
- `train.py`       trains on MNIST and saves `digits_classifier.pt`
- `preprocess.py`  converts a canvas drawing into an MNIST-style 28x28 input
- `app.py`         Gradio app with a drawing canvas

