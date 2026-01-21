````markdown
# rotate_anim.py

High-dimensional rotation → PCA → animated visualization, with **CLI export to GIF and MP4** (plus optional live preview).

It simulates a unit vector rotating in a random subspace of a high-dimensional space, then:
- projects the trajectory to 3D with PCA,
- shows the 3D path + current point,
- shows a cosine similarity matrix over time,
- shows the full “neuron × time” activity heatmap,
- exports the animation as a **portable** `.gif` or `.mp4`.

---

## Demo

- **GIF**: easy to share anywhere, typically larger file
- **MP4**: usually *much smaller* at comparable visual quality

---

## Requirements

### Python packages
- Python 3.9+
- `numpy`
- `matplotlib`
- `scikit-learn`
- `pillow` *(only for GIF export)*

Install (pip):

```bash
pip install numpy matplotlib scikit-learn pillow
````

### System dependency (for MP4)

MP4 export uses `ffmpeg`.

Ubuntu/Debian:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Conda:

```bash
conda install -c conda-forge ffmpeg
```

---

## Usage

### Show interactive animation window only

```bash
python rotate_anim.py --show
```

### Export GIF

```bash
python rotate_anim.py --gif rotation.gif
```

### Export MP4

```bash
python rotate_anim.py --mp4 rotation.mp4
```

### Export both in one run

```bash
python rotate_anim.py --gif rotation.gif --mp4 rotation.mp4
```

---

## Presets

`--preset` sets a recommended bundle of `fps / dpi / bitrate / frame-step` (and a preview-friendly `interval`).
You can still override any individual flag explicitly.

Available:

* `tiny` (small files)
* `balanced` (default-ish)
* `hq` (best quality, larger files)

Examples:

```bash
python rotate_anim.py --preset tiny --gif out.gif
python rotate_anim.py --preset balanced --mp4 out.mp4
python rotate_anim.py --preset hq --gif out.gif --mp4 out.mp4
```

Override example (explicit flags win):

```bash
python rotate_anim.py --preset tiny --mp4 out.mp4 --fps 24
```

---

## “~15 seconds, higher resolution” MP4 recipe

Duration is approximately:

```
duration_seconds ≈ (steps / frame_step) / fps
```

For ~15s at 30fps: `steps=450`, `frame_step=1` (450 frames / 30 fps = 15s)

```bash
python rotate_anim.py --mp4 rotation_15s_hires.mp4 \
  --steps 450 --frame-step 1 --fps 30 \
  --dpi 180 --bitrate 6000
```

---

## CLI Flags

### Export targets

* `--gif PATH` : write a GIF
* `--mp4 PATH` : write an MP4 (requires `ffmpeg`)
* `--show` : open a Matplotlib window (optional)

### Presets

* `--preset {tiny,balanced,hq}` : apply bundled export settings

### Simulation

* `--neurons INT` : dimensionality (default: 100)
* `--steps INT` : number of time steps / frames before subsampling (default: 200)
* `--seed INT` : RNG seed (default: 42)

### Animation/export control

* `--frame-step INT` : render every Nth step (default: 2). Higher → smaller files, shorter duration.
* `--fps INT` : export FPS (default: 20)
* `--dpi INT` : export DPI (default: 110). Higher → sharper, larger.
* `--bitrate INT` : MP4 bitrate (default: 1800). Higher → better quality, larger.
* `--interval INT` : preview interval (ms), mainly affects `--show` (default: 50)

---

## Notes & troubleshooting

### MP4 export fails

If you see an error about `ffmpeg`, verify it’s installed and on PATH:

```bash
ffmpeg -version
```

### GIF export fails

Install Pillow:

```bash
pip install pillow
```

### Files too large

Try one or more:

* increase `--frame-step` (e.g. 4)
* lower `--dpi` (e.g. 80–120)
* lower `--fps` (e.g. 12–20)
* lower MP4 `--bitrate` (e.g. 900–2500)

---

## License

MIT (or your preferred license — add a `LICENSE` file if you want this to be explicit).

```
```
