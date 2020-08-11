#!/Users/dguillen/opt/anaconda3/bin/python3

import cv2
import itertools
import tensorflow as tf

from dataset import *

def _test_helper_non_window(ds, stop_at_idx=5):
    idx = 1
    for frame, speed in ds:
        if idx >= stop_at_idx:
            break

        print(f"Frame {idx} has speed: {speed}")
        print(f"Frame {idx} has type: {frame.dtype}")
        print(f"Frame {idx} has shape: {frame.shape}")
        print()

        # Visualize the given frame.
        cv2.imshow("frame", frame.numpy())
        cv2.waitKey()

        idx = idx+1

def test_get_original_dataset():
    ds = get_original_dataset()

    _test_helper_non_window(ds)
        
def test_to_greyscale():
    ds = get_original_dataset()
    greyscale_ds = ds.map(to_greyscale)

    _test_helper_non_window(greyscale_ds)

def test_to_shrink_by_half_w_resize():
    ds = get_original_dataset()
    shrunk_ds = ds.map(shrink_by_half_w_resize)

    _test_helper_non_window(shrunk_ds)

def test_augment_image_frames():
    ds = get_original_dataset()
    aug_ds = augment_image_frames(ds)

    _test_helper_non_window(aug_ds)

def _test_helper_window(ds, stop_at_idx=5):
    idx = 1
    for stacked_frames, speeds in ds:
        if idx >= stop_at_idx:
            break

        _, _, window_size = stacked_frames.shape
        
        for frame_idx in range(window_size):
            frame = stacked_frames[:, :, frame_idx]
            speed = speeds[frame_idx]

            print(f"Frame {idx} has speed: {speed}")
            print(f"Frame {idx} has type: {frame.dtype}")
            print(f"Frame {idx} has shape: {frame.shape}")
            print(f"Frame {idx} is on channel: {frame_idx}")
            print()

            # Visualize the given frame.
            cv2.imshow("frame", frame.numpy())
            cv2.waitKey()

            idx = idx + 1


def test_merge_image_frames():
    ds = get_original_dataset()
    window_ds = merge_image_frames(ds, window_size=4)

    _test_helper_window(window_ds)

def test_get_pre_processed_dataset():
    ds = get_pre_processed_dataset()

    _test_helper_window(ds)

def _test_helper_run_to_the_end(ds):
    count = 1
    for inputs, labels in ds:

        if count % 1000 == 0:
            print(f"ds instance num: {count}")

        count += 1

    print(f"Finished iterating through dataset with {count} instances")

def test_pre_processed_dataset_to_end():
    ds = get_pre_processed_dataset()
    _test_helper_run_to_the_end(ds)
    
def test_original_dataset_to_end():
    ds = get_original_dataset()
    _test_helper_run_to_the_end(ds)

def main():
    test_to_greyscale()
    # test_get_original_dataset()
    # test_get_pre_processed_dataset()
    # test_pre_processed_dataset_to_end()

if __name__ == "__main__":
    main()
