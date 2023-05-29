import os
import torch
import torchaudio
import sox
import json
import random
import argparse


from text_normalization import text_normalize
from align_utils import (
    get_uroman_tokens,
    time_to_frame,
    load_model_dict,
    merge_repeats,
    get_spans,
)
import torchaudio.functional as F
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_emissions(model, audio_file):
    waveform, _ = torchaudio.load(audio_file)  # waveform: channels X T
    waveform = waveform.to(DEVICE)
    total_duration = sox.file_info.duration(audio_file)

    audio_sf = sox.file_info.sample_rate(audio_file)
    assert audio_sf == SAMPLING_FREQ

    emissions_arr = []
    with torch.inference_mode():
        i = 0
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)

            context = EMISSION_INTERVAL * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[
                :,
                int(SAMPLING_FREQ * input_start_time) : int(
                    SAMPLING_FREQ * (input_end_time)
                ),
            ]
            model_outs, _ = model(waveform_split)
            emissions_ = model_outs[0]
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)

            emissions_ = emissions_[
                emission_start_frame - offset : emission_end_frame - offset, :
            ]
            emissions_arr.append(emissions_)
            i += EMISSION_INTERVAL

    emissions = torch.cat(emissions_arr, dim=0).squeeze().detach()
    emissions = torch.log_softmax(emissions, dim=-1).detach()
    emissions = emissions.clone().detach()

    stride = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)

    return emissions, stride


def get_alignments(
    audio_file,
    tokens,
    model,
    dictionary,
    use_star,
):
    # Generate emissions
    emissions, stride = generate_emissions(model, audio_file)
    T, N = emissions.size()
    if use_star:
        emissions = torch.cat([emissions, torch.zeros(T, 1).to(DEVICE)], dim=1)

    # Force Alignment
    if tokens:
        token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    else:
        print(f"Empty transcript!!!!! for audio file {audio_file}")
        token_indices = []

    blank = dictionary["<blank>"]
    
    targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)
    input_lengths = torch.tensor(emissions.shape[0])
    target_lengths = torch.tensor(targets.shape[0])
    path, _ = F.forced_align(
        emissions, targets, input_lengths, target_lengths, blank=blank
    )
    path = path.to("cpu").tolist()
    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return segments, stride




def do_batch(model, dictionary, align_data, lang, use_star=True, cut=False, uroman_path="uroman/bin"):
    for index, ele in enumerate(align_data):
        transcripts, audio_filepath, outdir = ele['transcripts'], ele['audio_file'], ele['outdir']
        print(f"{index} / {len(align_data)} processing {audio_filepath} start.")

        os.makedirs(outdir, exist_ok=True)
        manifest_file = f"{outdir}/manifest.json"
        if os.path.exists(manifest_file):
            print(f"{manifest_file} already existed. skip it...")
            continue

        norm_transcripts = [text_normalize(line.strip(), lang) for line in transcripts]
        tokens = get_uroman_tokens(norm_transcripts, uroman_path, lang)
        if use_star:
            if "<star>" not in dictionary:
                dictionary["<star>"] = len(dictionary)
            stars = ["<star>"] * len(tokens)
            tokens = [i for pair in zip(tokens, stars) for i in pair]
            tokens = ["<star>"] + tokens
            transcripts = [i for pair in zip(transcripts, stars) for i in pair]
            transcripts = ["<star>"] + transcripts
            norm_transcripts = [i for pair in zip(norm_transcripts, stars) for i in pair]
            norm_transcripts = ["<star>"] + norm_transcripts
        segments, stride = get_alignments(
            audio_filepath,
            tokens,
            model,
            dictionary,
            use_star,
        )

        # Get spans of each line in input text file
        spans = get_spans(tokens, segments)

        with open(manifest_file, "w") as f:
            for i, t in enumerate(transcripts):
                span = spans[i]
                if transcripts[i] == "<star>":
                    continue
                seg_start_idx = span[0].start
                seg_end_idx = span[-1].end

                audio_start_sec = seg_start_idx * stride / 1000
                audio_end_sec = seg_end_idx * stride / 1000 
                
                sample = {
                    "audio_start_sec": audio_start_sec,
                    "audio_end_sec": audio_end_sec,
                    "duration": audio_end_sec - audio_start_sec,
                    "text": t,
                    "normalized_text":norm_transcripts[i],
                    "uroman_tokens": tokens[i],
                }
                if cut:
                    output_file = f"{outdir}/segment{i}.flac"
                    tfm = sox.Transformer()
                    tfm.trim(audio_start_sec , audio_end_sec)
                    tfm.build_file(audio_filepath, output_file)
                    sample["audio_filepath"] = str(output_file)
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"{index} / {len(align_data)} processing {manifest_file} done.")

def main(args):

    model, dictionary = load_model_dict()
    model = model.to(DEVICE)
    print("model has been loaded.")

    cons = open(args.info_filepath).readlines()
    align_data_list = []
    for line in cons:
        # <audio-name>\t<audio-path>\t<segmented-text1>\t...\t<segmented-textn>
        name, wavp, *transcripts = line.strip().split("\t")
        align_data_list.append(
            {
                "audio_file": wavp,
                "transcripts": transcripts,
                "outdir": os.path.join(args.outdir, name)
            }
        )
    random.shuffle(align_data_list)

    align_data_list = align_data_list
    num_threads = args.num_threads
    num_batch = len(align_data_list) // num_threads + 1
    batches = [ align_data_list[index*num_batch:(index+1)*num_batch] for index in range(num_threads)]

    tasks = []
    with ThreadPoolExecutor(max_workers=num_threads) as t:
        for batch in batches:
            task = t.submit(do_batch, model, dictionary, batch, args.lang, args.use_star, args.cut)
            tasks.append(task)
            
        wait(tasks, return_when=ALL_COMPLETED)
    
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align and segment long audio files")
    parser.add_argument(
        "-i", "--info_filepath", type=str, help="Path to information for waited segmented files."
    )
    parser.add_argument(
        "-l", "--lang", type=str, default="eng", help="ISO code of the language"
    )
    parser.add_argument(
        "-s", "--use_star", type=bool, default=True, help="Use star at the start of transcript",
    )
    parser.add_argument(
        "-o", "--outdir", type=str, help="Output directory to store segmented results, maybe the segmented files.",
    )
    parser.add_argument(
        "-c", "--cut", type=bool, default=False, help="Whether cut the long audio to small pieces according to the alignment results.",
    )
    parser.add_argument(
        "-t", "--num_threads", type=int, default=25, help="number of threads to do the alignment.",
    )
    args = parser.parse_args()
    main(args)
