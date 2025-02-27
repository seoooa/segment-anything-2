# segment-anything-2
Run Segment Anything Model 2 on a **static image**, **video** and **live video stream**

## Demo
<div align=center>
<p align="center">

### For static image
<img src="./assets/image_mask.jpg" width="880">
</p>

### For video
<img src="./assets/video_mask.gif" width="880">
</p>

### For real-time video stream
<img src="./assets/real_time_mask.gif" width="880">
</p>
</div>


## Getting Started

### Installation

```bash
pipenv shell
pipenv install
```
### Download Checkpoint

Then, we need to download a model checkpoint.

```bash
# excute on git bash
cd checkpoints
./download_ckpts.sh

```
