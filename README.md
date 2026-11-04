<h1 align="center">LiveProBench</h1>
<h3 align="center">Can Streaming Video Models Really Interact Like Humans?</h3>

<p align="center">
  <a href="https://github.com/v0yager33/LiveProBench"><img src="https://img.shields.io/badge/GitHub-Code-181717?style=flat-square&amp;logo=github" alt="GitHub Code"></a>
  <a href="https://arxiv.org/abs/2609.12658"><img src="https://img.shields.io/badge/arXiv-2609.12658-B31B1B?style=flat-square&amp;logo=arxiv" alt="arXiv Paper"></a>
  <a href="https://huggingface.co/datasets/v0yager33/LiveProBench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E?style=flat-square" alt="Hugging Face Dataset"></a>
  <a href="https://modelscope.cn/datasets/Voyager33/LiveProBench"><img src="https://img.shields.io/badge/ModelScope-Dataset-624AFF?style=flat-square" alt="ModelScope Dataset"></a>
</p>

<p align="center">
  <b>6 subtasks</b> &nbsp;·&nbsp; <b>911 test samples</b> &nbsp;·&nbsp; <b>1 fps evaluation</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> &nbsp;|&nbsp;
  <a href="#official-dataset">Dataset</a> &nbsp;|&nbsp;
  <a href="#evaluation">Evaluation</a> &nbsp;|&nbsp;
  <a href="#citation">Citation</a>
</p>

---

## Overview

**LiveProBench** evaluates **when to speak—and when to stay silent** in streaming video interaction. A model monitors a standing request as the stream unfolds in one-second intervals, without an explicit cue telling it when to respond. The benchmark measures timely responses, premature speech, missed events, and repeated responses.

Its six subtasks vary **trigger ambiguity** and **timing tolerance**:

| Subtask | Evaluation focus |
| :--- | :--- |
| **Event Sensitivity (ES)** | Respond to the first evidence of an event while remaining silent beforehand. |
| **Onset Response (OR)** | Respond promptly at the onset of the requested event. |
| **Window Response (WR)** | Respond within the annotated time window. |
| **Anomaly Alert (AA)** | Respond to semantically fuzzy terms like "anomaly" within the window. |
| **Sustained Silence (SS)** | Remain silent through a long prefix until the requested event occurs. |
| **Duplicate Counting (DC)** | Count each occurrence without omissions or duplicate reports. |

## Official dataset

Download the official dataset from [Hugging Face](https://huggingface.co/datasets/v0yager33/LiveProBench) or [ModelScope](https://modelscope.cn/datasets/Voyager33/LiveProBench). It contains six public test subtasks (911 samples), source videos split into 13 archive parts (`aa` through `am`), and `make_video_chunks.py`.

From the downloaded dataset directory:

```bash
cat src/videos.tar.part_* | tar -xf -
python3 make_video_chunks.py --annotations test --workers 12
```

Video preprocessing requires Python 3.7+, `ffmpeg`, and `ffprobe`. Download all 13 parts before extraction. The script generates the `video_chunks/` frame/audio tree; it is not a pre-extracted download.

## Evaluation

The default benchmark evaluation protocol is **1 fps**: one full-resolution frame per 1-second chunk (`--frames-per-chunk 1 --frame-max-side 0`). Use this configuration for standard benchmark results. Other sampling rates are non-default experiments and must be reported separately.

Install dependencies and run from this code repository with a compatible OpenAI-style model endpoint:

```bash
pip install -r requirements.txt
python3 src/run_eval.py \
  --model LiveFlow_omni \
  --base-url http://localhost:8000/v1 \
  --model-id YOUR_SERVED_MODEL_ID \
  --suite all \
  --layout test \
  --data-root /path/to/downloaded/LiveProBench \
  --media-root /path/to/downloaded/LiveProBench/video_chunks \
  --frames-per-chunk 1 \
  --frame-max-side 0 \
  --workers 4
```

`--data-root` is the downloaded dataset directory, not this code repository. The evaluator accepts the public `test/` schema and the generated `video_chunks/` layout, reading one middle frame per chunk under the default 1 fps protocol. Model endpoints must be able to access these local media files. For other backends and options, run `python3 src/run_eval.py --help`.

## Citation

If you find this benchmark useful in your research, please consider citing **LiveProBench** and our related work, **Live-OPSD**. Thank you for supporting our work!

- [LiveProBench: Can Streaming Video Models Really Interact Like Humans?](https://arxiv.org/abs/2609.12658)
- [Live-OPSD: On-Policy Self-Distillation for Proactive Streaming Large Video Models](https://arxiv.org/abs/2610.36959)

```bibtex
@misc{du2026liveprobench,
  title={{LiveProBench}: Can Streaming Video Models Really Interact Like Humans?},
  author={Kaixuan Du and Xin Wan and Hang Zhang and Meng Cao and Dai Guan and Ming Chen and YuKun Wang},
  year={2026},
  eprint={2609.12658},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2609.12658}
}

@misc{du2026liveopsd,
  title={{Live-OPSD}: On-Policy Self-Distillation for Proactive Streaming Large Video Models},
  author={Kaixuan Du and Xin Wan and Dai Guan and Ming Chen and Xiaoxi Jiang and Guanjun Jiang},
  year={2026},
  eprint={2610.36959},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2610.36959}
}
```
