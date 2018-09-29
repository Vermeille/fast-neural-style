# Fast Style Transfer

A Fast Style Transfer based on Pytorch.

# Usage

Train a model:

```
python3 fast_style.py --style a_style_image.jpg --trainimg training_dir/
```

where `--trainimg` expects a directory containing one or many subdirectories
containing many images (to comply with torch's ImageFolder). It generates
`style_model.pth`


Convert a video:

```
python3 style2vid.py --input input.avi --output result.avi --outsize 1920,1080
```

where `--input` is an input video, and `--outsize` is twice the size of a frame
in the input video.

# Demo

[This video](https://www.youtube.com/watch?v=TYgIBV_C-SY) was made using this
code. Note that the effects during the colored parts are not done through style
transfer. The raw result can be seen during the black and white parts.
