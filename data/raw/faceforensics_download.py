#!/usr/bin/env python
"""Download the authorized FaceForensics v1 public data release.

The dataset is distributed under the FaceForensics terms of use. This script
is intentionally kept close to the authors' release script so its URLs and
dataset layout remain auditable.
"""

import argparse
import os
import tempfile
import urllib.request
from os.path import join

SERVER_URL = "http://kaldir.vc.in.tum.de/FaceForensics/"
TOS_URL = SERVER_URL + "webpage/FaceForensics_TOS.pdf"
BASE_URL = SERVER_URL + "v1_cargo/"
ORIGINAL_VIDEOS_URL = BASE_URL + "original_videos.tar.gz"
RELEASE_DATASET_SIZE = {"raw": "~3.5TB", "compressed": "~130GB", "images": "~135GB"}
DATASET_TYPES = [
    "raw",
    "compressed",
    "selfreenactment_raw",
    "selfreenactment_compressed",
    "original_videos",
    "selfreenactment_images",
    "source_to_target_images",
]
NUM_SAMPLES = 5


def get_filelist(filelist_url):
    with urllib.request.urlopen(filelist_url) as response:
        return [line.decode("utf-8").rstrip("\n") for line in response]


def download_file(url, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if os.path.isfile(out_file):
        print("WARNING: skipping existing file " + out_file)
        return
    handle, temporary_path = tempfile.mkstemp(dir=os.path.dirname(out_file))
    os.close(handle)
    try:
        urllib.request.urlretrieve(url, temporary_path)
        os.replace(temporary_path, out_file)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def download_files(filenames, base_url, output_path, sample_only=False):
    os.makedirs(output_path, exist_ok=True)
    selected = filenames[:NUM_SAMPLES] if sample_only else filenames
    for index, filename in enumerate(selected, 1):
        print(f"{index}/{len(selected)}")
        download_file(base_url + filename, join(output_path, filename))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_path", help="directory in which to download")
    parser.add_argument(
        "-d",
        "--dataset_type",
        default="compressed",
        choices=DATASET_TYPES,
    )
    parser.add_argument("--not_altered", action="store_true")
    parser.add_argument("--not_original", action="store_true")
    parser.add_argument("--not_mask", action="store_true")
    parser.add_argument("--not_test", action="store_true")
    parser.add_argument("--not_train", action="store_true")
    parser.add_argument("--not_val", action="store_true")
    parser.add_argument("--sample_only", action="store_true")
    args = parser.parse_args()

    print("By continuing you confirm that you accepted the FaceForensics terms:")
    print(TOS_URL)
    input("Press Enter to continue, or Ctrl-C to exit. ")

    video_types = []
    if not args.not_altered:
        video_types.append("altered")
    if not args.not_original:
        video_types.append("original")
    if "images" not in args.dataset_type and not args.not_mask:
        video_types.append("mask")

    folders = []
    if not args.not_test:
        folders.append("test")
    if not args.not_train:
        folders.append("train")
    if not args.not_val:
        folders.append("val")

    if "selfreenactment" in args.dataset_type:
        dataset = "selfreenactment"
        dataset_type = args.dataset_type.replace("selfreenactment_", "")
    else:
        dataset = "source_to_target"
        dataset_type = args.dataset_type.replace("source_to_target_", "")

    if args.dataset_type != "original_videos":
        print(f"Requested release size: {RELEASE_DATASET_SIZE[dataset_type]}")
        if not args.sample_only:
            print("WARNING: this may require substantial disk space.")
        input("Press Enter to continue, or Ctrl-C to exit. ")

    if args.dataset_type == "original_videos":
        download_file(
            ORIGINAL_VIDEOS_URL,
            join(args.output_path, "faceforensics_original_videos.tar.gz"),
        )
        return

    for folder in folders:
        filelist_folder = "images_" + folder if "images" in args.dataset_type else folder
        filelist_url = f"{BASE_URL}{dataset}/filelists/{filelist_folder}.txt"
        filenames = get_filelist(filelist_url)
        for video_type in video_types:
            output_path = join(
                args.output_path,
                f"FaceForensics_{args.dataset_type}",
                folder,
                video_type,
            )
            base_url = f"{BASE_URL}{dataset}/{dataset_type}/{folder}/{video_type}/"
            download_files(
                filenames,
                base_url,
                output_path,
                sample_only=args.sample_only,
            )


if __name__ == "__main__":
    main()
