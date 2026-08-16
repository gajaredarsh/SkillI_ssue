# Trained model weights go here

Place your final trained checkpoint in this folder as:

    weights/scunet_finetuned.pth

This is the default path `evaluate.py` looks for (override with `--weights <path>` if
you name it differently or host it elsewhere).

If the file is large, per the submission requirements you can either:
- Use Git LFS to track it (`git lfs track "weights/*.pth"`) and commit normally, or
- Host it externally (Google Drive / Hugging Face) and put the download link here,
  with a one-line `wget`/`gdown` command in the main README's setup instructions so
  a reviewer can fetch it without contacting you.

Delete this placeholder file once the real weights file is in place.
