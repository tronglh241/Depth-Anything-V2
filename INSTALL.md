# Depth Anything V2 Installation Guide

This guide will walk you through setting up the environment to run **Depth Anything V2**.

### 1. Download Model Checkpoints

Before running the model, you'll need to download the required checkpoints. Follow these steps:

1. Create a directory called `checkpoints` to store the downloaded files:
   ```bash
   mkdir checkpoints
   ```

2. Download the specific model checkpoints into the `checkpoints` directory:
   ```bash
   wget -P checkpoints https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
   wget -P checkpoints https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
   wget -P checkpoints https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
   ```

### 2. Install Required Dependencies

#### 2.1 Install PyTorch

To run Depth Anything V2, you must first install the appropriate version of PyTorch. Use the following command to install PyTorch and torchvision:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

#### 2.2 Install Other Dependencies

Next, install the additional dependencies required by Depth Anything V2. You can do this by running:

```bash
pip install -r requirements.txt
```

This will ensure that all necessary libraries are installed.

### 3. Run Depth Anything V2

After completing the setup, you can now run **Depth Anything V2** with the installed environment and downloaded checkpoints.
