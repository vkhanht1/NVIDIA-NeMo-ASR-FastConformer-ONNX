# 🎙️ NVIDIA NeMo ASR: FastConformer Training & ONNX Deployment

This project demonstrates a complete end-to-end pipeline for building an Automatic Speech Recognition (ASR) system using the **NVIDIA NeMo** framework. It covers everything from environment setup and configuration troubleshooting to training a **FastConformer** model on the `an4` dataset and optimizing performance via **ONNX Runtime** export.

## 🚀 Key Features & Skills Demonstrated
* **Framework Proficiency:** Utilized **NVIDIA NeMo Toolkit** for audio data processing and training a CTC-based Large FastConformer model.
* **Advanced Configuration:** Managed complex model parameters using **OmegaConf**, resolving deep-link variable issues such as `InterpolationKeyError`.
* **Session & Environment Management:** Handled library version conflicts between `pytorch-lightning` and `nemo_toolkit`, ensuring a synchronized Google Colab environment.
* **Deployment Optimization:** 
    * Exported PyTorch models to the high-performance `.onnx` format.
    * Implemented Inference using **ONNX Runtime** for accelerated processing on both GPU and CPU.
    * Managed multi-variable input streams (`audio_signal` and `length`) required for complex ASR ONNX models.

## 🛠️ Project Structure
* `NVIDIA_NeMo_ASR_FastConformer_ONNX.ipynb`: The main Notebook containing the full workflow from training to export.
* `fastconformer.onnx`: The optimized version of the model ready for production deployment.
* `README.md`: Project documentation and guides.

## ⚙️ Installation & Setup
To run this project in a local or Colab environment, install the following dependencies:

```bash
pip install nemo_toolkit[all]
pip install onnxruntime-gpu
pip install pytorch-lightning
