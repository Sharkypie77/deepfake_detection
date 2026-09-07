# Datasets and model artifacts

This directory intentionally does not contain training datasets. Downloading
and redistributing them requires accepting each provider's terms.

## ASVspoof LA

AASIST was published and evaluated for the ASVspoof 2019 Logical Access (LA)
track. Register with the ASVspoof/Edinburgh DataShare release, download the
LA package, and place it outside version control, for example:

```text
data/asvspoof2019/LA/
  ASVspoof2019_LA_train/flac/
  ASVspoof2019_LA_dev/flac/
  ASVspoof2019_LA_eval/flac/
  ASVspoof2019_LA_cm_protocols/
```

The dataset is provided for research under its own access and usage terms;
registration is required. See https://www.asvspoof.org/ and
https://datashare.ed.ac.uk/handle/10283/3336.

## Vision datasets

FaceForensics++ access has been granted for this project. The script supplied
for this checkout is the **legacy FaceForensics v1** release script, not the
FaceForensics++ v4 script. It is saved as
`data/raw/faceforensics_download.py`. Download the official script from
https://kaldir.vc.in.tum.de/faceforensics_download_v4.py and accept the
FaceForensics terms when prompted. The current available server is EU2:

```powershell
python data/raw/faceforensics_download.py data/raw/faceforensics `
  --dataset_type compressed --sample_only
```

The supplied v1 script does not provide the FF++ `c23` option or the FF++
Deepfakes/Face2Face/FaceSwap/NeuralTextures layout. For the requested FF++
training pipeline, use the approved v4 script/link from the dataset authors,
run it with `--server EU2`, and download `--compression c23` for
`original`, `Deepfakes`, `Face2Face`, `FaceSwap`, and `NeuralTextures`.
Use the v1 script above only for a legacy FaceForensics smoke sample. A
production training run needs the complete, authorized train/validation/test
split; do not treat a small sample as a validated model.

For the existing fine-tuning loader, organize or link the downloaded files
into this class-oriented layout (original videos are authentic; each
manipulation directory is deepfake):

```text
data/training/
  authentic/faceforensics/*.mp4
  deepfake/deepfakes/*.mp4
  deepfake/face2face/*.mp4
  deepfake/faceswap/*.mp4
  deepfake/neuraltextures/*.mp4
```

Follow the license and access terms of the selected dataset and do not commit media:

```text
data/training/
  authentic/<language-or-source>/*.mp4
  deepfake/<manipulation-method>/*.mp4
```

FaceForensics++ requires an academic/research request and its terms restrict
redistribution. DFDC and Celeb-DF also have dataset-specific terms. Add
regional authentic and synthetic samples only when the creator and licensing
rights permit it.

## Checkpoints

The official MIT-licensed Clova AASIST checkpoint may be downloaded from
https://github.com/clovaai/aasist/blob/main/models/weights/AASIST.pth. The
repository can load it, but it is **not marked validated here**: the published
ASVspoof EER is not a measurement performed by this checkout. Create a sidecar
manifest only after running the repeatable evaluation script on an authorized
held-out set.
