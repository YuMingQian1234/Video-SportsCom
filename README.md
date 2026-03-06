VideoSportsCom: Video-SportsCom: A Large-Scale Sports Commentary Dataset for Artistic, Technical, and Tactical Sports Video Understanding
## Introduction
VideoSportsCom is large-scale, multi-category, multi-attribute, and multi-sport dataset for artistic, technical, and tactical sports video understanding.


## Requirements and Installation

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
git clone https://github.com/YuMingQian1234/Video-SportsCom
cd Video-SportsCom
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
            {"type": "video", "video": {"video_path": "./assets/cat_and_chicken.mp4", "fps": 15, "max_frames": 300}},
            {"type": "text", "text": "You are a professional commentator. Provide color commentary with a positive stance and a non-passionate style on the given video."},
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





##  Training

### Step 1: Prepare training data
To use our training code, please organize the image and video data as you like under `data_root`, and then use one or more annotation files to record each conversation data and the corresponding image/video path. For example:
```bash
data_root
├── train
│   ├── video_1.mp4
│   └── ...
│   └── ...
├── Video-SportsCom_Train.jsonl
├── Video-SportsCom_Train_Masked.jsonl
└── ...
```
The annotation files are consist of a list of dictionaries, where each item follows the following format:
```json
[
    {
        "video": ["videos/xxx.mp4"],
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nYou are a professional commentator. Provide color commentary with a positive stance and a non-passionate style on the given video."
            },
            {
                "from": "gpt",
                "value": "Yes and I think the direct comparison as well when you see everyone do their clubs and then everyone do their hoop and ribbon and so on also gives the judges a better perspective because in qualification you're alternating apparatus between two subgroups."
            },
            ...
        ]
    },
    ...
]
```
For loading and memory efficiency, we recommend to use `.jsonl` files with [huggingface datasets](https://huggingface.co/docs/datasets/loading) format.
### Step 3: Prepare training script
We provide some templates in `scripts/train` for all stages. You can modify the variables to fit your settings of data and models based on them. For example:
```bash
  --data_folder ./datasets \
  --data_path ./datasets/Video-SportsCom_Train.jsonl ./datasets/Video-SportsCom_Train_Masked.jsonl \
  --model_path DAMO-NLP-SG/VideoLLaMA3-7B \
  --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
```
For finetuneing, `--model_path` is the path to the converted checkpoint as described in step 2.
### Step 4: Start training
Now you can start training with your training scripts:
```bash
bash scripts/train/stage1_2b.sh
```
### Some tips about CUDA OOM error:
- Please try the latest main branch, where we optimize the memory consumption in [this commit](https://github.com/DAMO-NLP-SG/VideoLLaMA3/commit/21268660a67c115c6d6c6620780515626193af0f).
- Try DeepSpeed [ZeRO-2/3](https://huggingface.co/docs/transformers/deepspeed) by passing `--deepspeed scripts/zero2.json / zero3.json`.
- Reduce the max number of visual tokens (high-resolution images and videos will be automatically downsampled to fit this length) and max length of sequences (sequences longer than this will be truncated) by setting `--mm_max_length` and `--model_max_length`, respectively.
- Reduce the local batch size, i.e., `LOCAL_BATCH_SIZE` in the training script.
You can adjust the above hyperparameters according to the available GPU memory and number of GPUs to make the training fits your hardware.
- **(New!)** If you still encounter memory issues after using the above tricks, you can try using an **experimental** feature by setting `--use_flash_loss True` in your training script. Specifically, it uses a tile-based CE implementation proposed in [Inf-CL](https://github.com/DAMO-NLP-SG/Inf-CLIP) to reduce the memory consumption, which is very helpful when training models with long context or large vocabulary!




## Citation

If you find Video-Sports useful for your research and applications, please cite using this BibTeX:

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

## Acknowledgement
Our Video-SportsCom is built on top of [**SigLip**](https://huggingface.co/google/siglip-so400m-patch14-384) and [**Qwen2.5**](https://github.com/QwenLM/Qwen2.5). We also learned a lot from the implementation of [**LLaVA-OneVision**](https://github.com/LLaVA-VL/LLaVA-NeXT), [**InternVL2**](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/), and [**Qwen2VL**](https://github.com/QwenLM/Qwen2-VL). Besides, our VideoLLaMA3 benefits from tons of open-source efforts. We sincerely appreciate these efforts and compile a list in [ACKNOWLEDGEMENT.md](https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/ACKNOWLEDGEMENT.md) to express our gratitude. If your work is used in VideoLLaMA3 but not mentioned in either this repo or the technical report, feel free to let us know :heart:.


## License

This project is released under the Apache 2.0 license as found in the LICENSE file.
The service is a research preview intended for **non-commercial use ONLY**, subject to the model Licenses of Qwen, Terms of Use of the data generated by OpenAI and Gemini, and Privacy Practices of ShareGPT. Please get in touch with us if you find any potential violations.
