# Description

> **This repository uses alignment tools from [MMS](https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/). Most of the code(99.99%) from [data_prep](https://github.com/facebookresearch/fairseq/tree/main/examples/mms/data_prep) of the original repository [fairseq](git@github.com:facebookresearch/fairseq.git).**

**What this repo DO:** Just re-organize the original code to get a pure-functional alignment tools for 1000+ languages.

* [X] **get more accurate result by adding **`<star>`** into the token, it get significant better results.**
  **I just changed from:**

  ```
    if args.use_star:
        dictionary["<star>"] = len(dictionary)
        tokens = ["<star>"] + tokens
        transcripts = ["<star>"] + transcripts
        norm_transcripts = ["<star>"] + norm_transcripts
  ```

  To:

  ```
  if args.use_star:
      dictionary["<star>"] = len(dictionary)
      stars = ["<star>"] * len(tokens)
      tokens = [i for pair in zip(tokens, stars) for i in pair]
      tokens = ["<star>"] + tokens
      transcripts = [i for pair in zip(transcripts, stars) for i in pair]
      transcripts = ["<star>"] + transcripts
      norm_transcripts = [i for pair in zip(norm_transcripts, stars) for i in pair]
      norm_transcripts = ["<star>"] + norm_transcripts
  ```

  and also add a filter to get rid of the **`<star>`** in the final results, line 141 in align_and_segment:

  ```
  if span == "<star>":
      continue
  ```

  The comparision will be added soon with `<star>` or not to show the change.
* [ ] **get this to be more handy to use by alignment a long audio or just align the words.**
* [ ] **support more input format and output format, may .wav, .mp3, or maybe just do'nt cut.**
* [ ] **thinking how to make the language more handy**
* [ ] **how to make this more automatically**

## Enviroments and Model Download

**We describe the process of aligning long audio files with their transcripts and generating shorter audio segments below.**

* **Get the repository**

```
git clone --recursive https://github.com/MXuer/mms-alignment-tools.git
```

* **Download and install torchaudio using the nightly version**[torchaudio](https://github.com/pytorch/audio/pull/3348).

  ```
  pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
  ```
* **Install dependecies**

  ```
  pip install -r requirements.txt
  ```
* **Download the model**

  ```
  wget -P align_model https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt
  wget -P align_model https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/dictionary.txt
  ```

# Processing multithread

- get a text file with a format `<audio-name>`\t `<audio-path>`\t `<segmented-text1>`\t...\t `<segmented-textn>`

  ```shell
  audio	/data/audio.wav	what	a	nice	data
  ```
- run the script file: `align_and_segment_multi.py` with

  ```
  python align_and_segment_multi.py -i ../preprocess_c/info.txt -l cmn -o ../preprocess_c/outputs -t 24
  ```
- it will give a result in `output directory` with a `manifest.json` contains the alignment information

# Data Preparation

* **Step 4: Create a text file containing the transcript for a (long) audio file. Each line in the text file will correspond to a separate audio segment that will be generated upon alignment.**
  **Example content of the input text file :**

  ```
  Text of the desired first segment
  Text of the desired second segment
  Text of the desired third segment
  ```
* **Step 5: Run forced alignment and segment the audio file into shorter segments.**

  ```
  python align_and_segment.py --audio /path/to/audio.wav --textfile /path/to/textfile --lang <iso> --outdir /path/to/output --uroman /path/to/uroman/bin 
  ```
  **The above code  will generated the audio segments under output directory based on the content of each line in the input text file. The **`manifest.json` file consisting of the of segmented audio filepaths and their corresponding transcripts.

  ```
  > head /path/to/output/manifest.json 

  {"audio_start_sec": 0.0, "audio_filepath": "/path/to/output/segment1.flac", "duration": 6.8, "text": "she wondered afterwards how she could have spoken with that hard serenity how she could have", "normalized_text": "she wondered afterwards how she could have spoken with that hard serenity how she could have", "uroman_tokens": "s h e w o n d e r e d a f t e r w a r d s h o w s h e c o u l d h a v e s p o k e n w i t h t h a t h a r d s e r e n i t y h o w s h e c o u l d h a v e"}
  {"audio_start_sec": 6.8, "audio_filepath": "/path/to/output/segment2.flac", "duration": 5.3, "text": "gone steadily on with story after story poem after poem till", "normalized_text": "gone steadily on with story after story poem after poem till", "uroman_tokens": "g o n e s t e a d i l y o n w i t h s t o r y a f t e r s t o r y p o e m a f t e r p o e m t i l l"}
  {"audio_start_sec": 12.1, "audio_filepath": "/path/to/output/segment3.flac", "duration": 5.9, "text": "allan's grip on her hands relaxed and he fell into a heavy tired sleep", "normalized_text": "allan's grip on her hands relaxed and he fell into a heavy tired sleep", "uroman_tokens": "a l l a n ' s g r i p o n h e r h a n d s r e l a x e d a n d h e f e l l i n t o a h e a v y t i r e d s l e e p"}
  ```
  **To visualize the segmented audio files, **[Speech Data Explorer](https://github.com/NVIDIA/NeMo/tree/main/tools/speech_data_explorer) tool from NeMo toolkit can be used.

  **As our alignment model outputs uroman tokens for input audio in any language, it also works with non-english audio and their corresponding transcripts.**
