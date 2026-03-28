# Speech Emotion Recognition Project Tasks

## 1. Feature Extraction Enhancements
- [ ] Add extraction of 13 MFCCs + deltas.
- [ ] Add extraction of chromagrams.
- [ ] Combine MFCCs, deltas, chromagrams, and mel-spectrograms into a unified feature set or adjust model to handle them as described in the requirements.

## 2. Model & Training Adjustments
- [ ] Add class weights for imbalance to the loss function (currently using standard `NLLLoss`).
- [ ] Add F1-score evaluation per emotion (macro average >0.80) to the Lightning module (currently only using Accuracy).
- [ ] Replace or supplement TensorBoard with MLflow for metrics logging.

## 3. Alternative Model Implementation
- [ ] Add Hugging Face `transformers` to dependencies.
- [ ] Implement Wav2Vec2 fine-tuning (add emotion head, sequence classification).

## 4. Deployment & Real-Time Inference
- [ ] Add `gradio`, `sounddevice`, and potentially `fastapi`/`uvicorn` (for API endpoint) to [pyproject.toml](file:///home/moad/Desktop/project/python/audio-project/pyproject.toml).
- [ ] Create an interactive Gradio demo for live emotion detection from mic input.
- [ ] Implement real-time pipeline to predict emotions and visualize the confusion matrix.
- [ ] Build API endpoint for the model.

## 5. Expected Deliverables & Documentation
- [ ] Complete Jupyter notebooks for EDA, features, and evaluation.
- [ ] Upload trained model to Hugging Face Hub.
- [ ] Complete [README.md](file:///home/moad/Desktop/project/python/audio-project/README.md) and generate `requirements.txt` / provide [uv.lock](file:///home/moad/Desktop/project/python/audio-project/uv.lock).
- [ ] Write the 20-page report (Model architecture, per-emotion accuracy, ablation on features/datasets).
- [ ] Record 3-minute video demo analyzing sample speeches (French/Darija).
