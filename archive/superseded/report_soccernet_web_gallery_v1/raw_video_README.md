---
license: other
license_link: https://drive.google.com/file/d/1Efz8yP-baa8CtcMB7SYIID7Bn5sH7sGTB4-u1YdVRf4/view
tags:
- soccer
- football
- sports
- video
- action-spotting
- temporal-action-localization
- computer-vision
- broadcast-video
pretty_name: SoccerNet Videos
extra_gated_heading: Request access to Dataset
extra_gated_description: >
  Complete the information below to request access. The SoccerNet team will
  review your request and the information you provide. If your request is
  accepted, a personalized Non-Disclosure Agreement will be sent to the email
  address associated with your Hugging Face account.
extra_gated_button_content: Sign the NDA and request access
extra_gated_prompt: >
  By submitting this request, you confirm that the information supplied is
  accurate and that you have read and accept the SoccerNet Non-Disclosure
  Agreement available at
  https://drive.google.com/file/d/1Efz8yP-baa8CtcMB7SYIID7Bn5sH7sGTB4-u1YdVRf4/view.
extra_gated_fields:
  Full Name: text
  Affiliation: text
  Please describe what you plan to do with the dataset: text
  I understand that the use of this dataset is for research and non-commercial use only: checkbox
  I understand that this dataset and the videos are protected by copyright: checkbox
  I agree not to share the dataset with anyone else: checkbox
  I agree to and electronically sign the Non-Disclosure Agreement: checkbox
---

# SoccerNet Tracking — Raw Single-Camera Video

Raw single-camera broadcast footage used in the SoccerNet Player Tracking benchmark, covering the 12 single-camera games.

**Gated** — request access on this page.

| File | Description |
|---|---|
| `single_camera_games_public.zip` | Raw single-camera video for all 12 tracking games |

## Download

Using the [SoccerNet pip package](https://pypi.org/project/SoccerNet/) (recommended):

```python
from SoccerNet.Downloader import SoccerNetDownloader
d = SoccerNetDownloader(LocalDirectory="path/to/soccernet")
d.downloadRAWVideo(dataset="SoccerNet-Tracking")
```

Directly with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="SoccerNet/SoccerNet-Tracking-RAW-Video",
    repo_type="dataset",
    local_dir="path/to/soccernet",
)
```

Annotations and the development kit for the tracking benchmark are in [`SoccerNet/SN-Tracking-2023`](https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023) and the [SoccerNet GitHub organization](https://github.com/SoccerNet).
