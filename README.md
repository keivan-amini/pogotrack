# Pogotrack

![Pogotrack Logo](logo.png)

Minimal tracking pipeline for Pogobots in arena videos.

---

## Usage

Run the main script with:

```bash
python3 scripts/main.py --video data/example.mp4 --background data/bkg.bmp --output results/tracking.csv --config config/default.yaml
```

---

## Folder Structure

- `config/` — YAML config files  
- `data/` — Input videos and backgrounds  
- `results/` — Output CSV files  
- `scripts/` — Execution scripts  
- `src/` — Source code modules  

---

## Requirements

Python3.10+

For python packages, see _requirements.txt_

---

## License

TODO

# Submodules

## Pogotrack_gui

A minimalistic tool available with a GUI to help write the configuration file.  