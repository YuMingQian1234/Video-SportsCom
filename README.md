<p align="center">
    <img src="https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/assets/logo.png?raw=true" width="150" style="margin-bottom: 0.2;"/>
<p>

<h3 align="center"><a href="https://arxiv.org/pdf/2501.13106" style="color:#9C276A">
VideoLLaMA 3: Frontier Multimodal Foundation Models for Video Understanding</a></h3>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>


<h5 align="center">

[![hf_space](https://img.shields.io/badge/🤗-Image_Demo-9C276A.svg)](https://huggingface.co/spaces/lixin4ever/VideoLLaMA3-Image)
[![hf_space](https://img.shields.io/badge/🤗-Video_Demo-9C276A.svg)](https://huggingface.co/spaces/lixin4ever/VideoLLaMA3)
[![hf_checkpoint](https://img.shields.io/badge/🤗-Checkpoints-9C276A.svg)](https://huggingface.co/collections/DAMO-NLP-SG/videollama3-678cdda9281a0e32fe79af15) <br>
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/LICENSE) 
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FDAMO-NLP-SG%2FVideoLLaMA3&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitor&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub issues](https://img.shields.io/github/issues/DAMO-NLP-SG/VideoLLaMA3?color=critical&label=Issues)](https://github.com/DAMO-NLP-SG/VideoLLaMA3/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/DAMO-NLP-SG/VideoLLaMA3?color=success&label=Issues)](https://github.com/DAMO-NLP-SG/VideoLLaMA3/issues?q=is%3Aissue+is%3Aclosed)  <br>
[![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2501.13106)
[![arXiv](https://img.shields.io/badge/Arxiv-2501.13106-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2501.13106) 
</h5>

<details open><summary>💡 Some other multimodal-LLM projects from our team may interest you ✨. </summary><p>
<!--  may -->




</p></details>


## 📰 News



## 🌟 Introduction
VideoLLaMA 3 is a series of multimodal foundation models with frontier image and video understanding capacity.

<img src="assets/performance.png" style="max-width: 100%; height: auto;">

<details>
  <summary>💡Click here to show detailed performance on video benchmarks</summary>
  <img src="https://github.com/user-attachments/assets/118e7a56-0c3e-4132-b0b5-f516d0654338" style="max-width: 100%; height: auto;">
  <img src="https://github.com/user-attachments/assets/3524cefe-01d3-4031-8620-f85dc38e3d02" style="max-width: 100%; height: auto;">
</details>

<details>
  <summary>💡Click here to show detailed performance on image benchmarks</summary>
  <img src="assets/results_image_2b.png" style="max-width: 100%; height: auto;">
  <img src="assets/results_image_7b.png" style="max-width: 100%; height: auto;">
</details>

## 🛠️ Requirements and Installation

Basic Dependencies:

* Python >= 3.10
* Pytorch >= 2.4.0
* CUDA Version >= 11.8
* transformers >= 4.46.3

Install required packages:

**[Inference-only]**

For stable inference, install the following package versions:

```bash
# PyTorch and torchvision for CUDA 11.8
pip install torch==2.4.0 torchvision==0.19.0 --extra-index-url https://download.pytorch.org/whl/cu118

# Flash-attn pinned to a compatible version
pip install flash-attn==2.7.3 --no-build-isolation --upgrade

# Transformers and accelerate
pip install transformers==4.46.3 accelerate==1.0.1

# Video processing dependencies
pip install decord ffmpeg-python imageio opencv-python
```
> ⚠ **Note:** For CUDA 11.8 with `torch==2.4.0` and `torchvision==0.19.0`, use `flash-attn==2.7.3`.  
> If you are using a different Python or CUDA version, please check the [flash-attn releases](https://github.com/Dao-AILab/flash-attention/releases/) to select the compatible wheel. Using incompatible versions may break the setup.

**[Training]**

```bash
git clone https://github.com/DAMO-NLP-SG/VideoLLaMA3
cd VideoLLaMA3
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```


## 🤖 Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

device = "cuda:0"
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": "./assets/cat_and_chicken.mp4", "fps": 1, "max_frames": 180}},
            {"type": "text", "text": "What is the cat doing?"},
        ]
    },
]

inputs = processor(
    conversation=conversation,
    add_system_prompt=True,
    add_generation_prompt=True,
    return_tensors="pt"
)
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
output_ids = model.generate(**inputs, max_new_tokens=1024)
response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(response)
```

For more cases, please refer to [examples](https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/inference/example_videollama3.py).


## 🤗 Demo

It is highly recommended to try our [online demo](https://huggingface.co/spaces/lixin4ever/VideoLLaMA3) first.

Otherwise, you can launch a gradio app locally:

```bash
python inference/launch_gradio_demo.py --model-path DAMO-NLP-SG/VideoLLaMA3-7B

options:
  --model-path MODEL_PATH, --model_path MODEL_PATH
  --server-port SERVER_PORT, --server_port SERVER_PORT
  	Optional. Port of the model server.
  --interface-port INTERFACE_PORT, --interface_port INTERFACE_PORT
  	Optional. Port of the gradio interface.
  --nproc NPROC
  	Optional. Number of model processes.
```

## 🗝️ Training

### Step 1: Prepare training data
To use our training code, please organize the image and video data as you like under `data_root`, and then use one or more annotation files to record each conversation data and the corresponding image/video path. For example:
```bash
data_root
├── LLaVA-Video-178K
│   ├── video_1.mp4
│   └── ...
├── LLaVA-OneVision-Data
│   ├── image_1.jpg
│   └── ...
├── annotations_video.jsonl
├── annotations_image.jsonl
└── ...
```
The annotation files are consist of a list of dictionaries, where each item follows the following format:
```json
[
    {
        "image": ["images/xxx.jpg"],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat are the colors of the bus in the image?"
            },
            {
                "from": "gpt",
                "value": "The bus in the image is white and red."
            },
            ...
        ]
    },
    {
        "video": ["videos/xxx.mp4"],
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nWhat are the main activities that take place in the video?"
            },
            {
                "from": "gpt",
                "value": "The main activities that take place in the video are the preparation of camera equipment by a man, a group of men riding a helicopter, and a man sailing a boat through the water."
            },
            ...
        ]
    },
    ...
]
```
For loading and memory efficiency, we recommend to use `.jsonl` files with [huggingface datasets](https://huggingface.co/docs/datasets/loading) format.
### Step 2: (Optional) Convert HF checkpoint
If you want to finetune VideoLLaMA3 on your own data using this codebase, please first convert the checkpoints from huggingface to local format. For example:
```bash
python scripts/convert_hf_checkpoint.py --model_path DAMO-NLP-SG/VideoLLaMA3-7B --save_path weights/videollama3_7b_local
```
### Step 3: Prepare training script
We provide some templates in `scripts/train` for all stages. You can modify the variables to fit your settings of data and models based on them. For example:
```bash
  --data_folder ./datasets \
  --data_path ./datasets/annotations_video.jsonl ./datasets/annotations_image.jsonl \
  --model_path Qwen/Qwen2.5-1.5B-Instruct \
  --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
```
For finetuneing, `--model_path` is the path to the converted checkpoint as described in step 2.
### Step 4: Start training
Now you can start training with your training scripts:
```bash
# VideoLLaMA3 Stage 1
bash scripts/train/stage1_2b.sh
# VideoLLaMA3 Stage 2
bash scripts/train/stage2_2b.sh
```
### Some tips about CUDA OOM error:
- Please try the latest main branch, where we optimize the memory consumption in [this commit](https://github.com/DAMO-NLP-SG/VideoLLaMA3/commit/21268660a67c115c6d6c6620780515626193af0f).
- Try DeepSpeed [ZeRO-2/3](https://huggingface.co/docs/transformers/deepspeed) by passing `--deepspeed scripts/zero2.json / zero3.json`.
- Reduce the max number of visual tokens (high-resolution images and videos will be automatically downsampled to fit this length) and max length of sequences (sequences longer than this will be truncated) by setting `--mm_max_length` and `--model_max_length`, respectively.
- Reduce the local batch size, i.e., `LOCAL_BATCH_SIZE` in the training script.
You can adjust the above hyperparameters according to the available GPU memory and number of GPUs to make the training fits your hardware.
- **(New!)** If you still encounter memory issues after using the above tricks, you can try using an **experimental** feature by setting `--use_flash_loss True` in your training script. Specifically, it uses a tile-based CE implementation proposed in [Inf-CL](https://github.com/DAMO-NLP-SG/Inf-CLIP) to reduce the memory consumption, which is very helpful when training models with long context or large vocabulary!


## ✅ Evaluation
#### Step 1: Prepare evaluation data
First, please download the corresponding data according to the official instructions and organize it into the following format:
<details>
<summary>Click here to view the dataset directory organization</summary>

```bash
benchmarks
└── video
│   ├── activitynet_qa
│   │   ├── all_test
│   │   ├── test_a.json
│   │   └── test_q.json
│   ├── charades
│   │   ├── Charades_v1
│   │   └── charades_annotations_test-random_prompt.json
│   ├── egoschema
│   │   ├── good_clips_git
│   │   └── questions.json
│   ├── longvideobench
│   │   ├── lvb_val.json
│   │   ├── subtitles
│   │   └── videos
│   ├── lvbench
│   │   ├── video
│   │   └── video_info.meta.jsonl
│   ├── mlvu
│   │   ├── json
│   │   └── video
│   ├── mvbench
│   │   ├── json
│   │   └── video
│   ├── nextqa
│   │   ├── map_vid_vidorID.json
│   │   ├── NExTVideo
│   │   └── test.csv
│   ├── perception_test
│   │   ├── mc_question_test.json
│   │   └── videos
│   ├── tempcompass
│   │   ├── captioning
│   │   ├── caption_matching
│   │   ├── multi-choice
│   │   ├── videos
│   │   └── yes_no
│   ├── videomme
│   │   ├── subtitles
│   │   ├── test-00000-of-00001.parquet
│   │   └── videos
```

</details>

#### Step 2: Start evaluation
```bash
bash scripts/eval/eval_video.sh ${MODEL_PATH} ${BENCHMARKS} ${NUM_NODES} ${NUM_GPUS}
```
You can change the directory of benchmarks and outputs via `DATA_ROOT` and `SAVE_DIR` in the evaluation script. Please check the scripts for more detailed usage.

#### Step 3: Add new benchmark
Coming soon...


## 📑 Citation

If you find VideoLLaMA useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{damonlpsg2025videollama3,
  title={VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding},
  author={Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, Peng Jin, Wenqi Zhang, Fan Wang, Lidong Bing, Deli Zhao},
  journal={arXiv preprint arXiv:2501.13106},
  year={2025},
  url = {https://arxiv.org/abs/2501.13106}
}

@article{damonlpsg2024videollama2,
  title={VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs},
  author={Cheng, Zesen and Leng, Sicong and Zhang, Hang and Xin, Yifei and Li, Xin and Chen, Guanzheng and Zhu, Yongxin and Zhang, Wenqi and Luo, Ziyang and Zhao, Deli and Bing, Lidong},
  journal={arXiv preprint arXiv:2406.07476},
  year={2024},
  url = {https://arxiv.org/abs/2406.07476}
}

@article{damonlpsg2023videollama,
  title = {Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding},
  author = {Zhang, Hang and Li, Xin and Bing, Lidong},
  journal = {arXiv preprint arXiv:2306.02858},
  year = {2023},
  url = {https://arxiv.org/abs/2306.02858}
}
```

## 👍 Acknowledgement
Our VideoLLaMA3 is built on top of [**SigLip**](https://huggingface.co/google/siglip-so400m-patch14-384) and [**Qwen2.5**](https://github.com/QwenLM/Qwen2.5). We also learned a lot from the implementation of [**LLaVA-OneVision**](https://github.com/LLaVA-VL/LLaVA-NeXT), [**InternVL2**](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/), and [**Qwen2VL**](https://github.com/QwenLM/Qwen2-VL). Besides, our VideoLLaMA3 benefits from tons of open-source efforts. We sincerely appreciate these efforts and compile a list in [ACKNOWLEDGEMENT.md](https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/ACKNOWLEDGEMENT.md) to express our gratitude. If your work is used in VideoLLaMA3 but not mentioned in either this repo or the technical report, feel free to let us know :heart:.


## 🔒 License

This project is released under the Apache 2.0 license as found in the LICENSE file.
The service is a research preview intended for **non-commercial use ONLY**, subject to the model Licenses of Qwen, Terms of Use of the data generated by OpenAI and Gemini, and Privacy Practices of ShareGPT. Please get in touch with us if you find any potential violations.
