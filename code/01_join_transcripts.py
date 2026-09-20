"""Join the frozen V7 utterance list to the official ASVspoof/VCTK map."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterances", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    used = pd.read_csv(args.utterances, sep="\t", dtype=str)
    source = pd.read_csv(args.metadata, sep="\t", dtype=str)
    cols = ["ASVspoof_ID", "VCTK_ID", "TTS_text"]
    if not set(cols).issubset(source.columns):
        raise ValueError("Official metadata lacks the required ASVspoof/VCTK columns")
    result = used.merge(source[cols], left_on="file_id", right_on="ASVspoof_ID",
                        how="left", validate="one_to_one")
    if result.TTS_text.isna().any() or result.TTS_text.str.strip().eq("").any():
        raise ValueError("At least one included utterance has no transcript")
    result = result.rename(columns={"VCTK_ID": "vctk_id", "TTS_text": "transcript"})
    result = result.drop(columns="ASVspoof_ID")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
