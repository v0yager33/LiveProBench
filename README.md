Official repository for [LiveProBench](https://arxiv.org/pdf/2609.12658).


## Official dataset

Download the official dataset from [ModelScope](https://modelscope.cn/datasets/Voyager33/LiveProBench). It contains six public test subtasks (911 samples), source videos split into 13 archive parts (`aa` through `am`), and `make_video_chunks.py`.

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