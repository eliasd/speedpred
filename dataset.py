#!/Users/dguillen/opt/anaconda3/bin/python3

import cv2
import itertools
import tensorflow as tf

# Constants for Comma.ai's speed prediction 
# training set.
#   - Original video data has a shape (480, 640, 3) where
#     HEIGHT=480, WIDTH=640, CHANNELS=3 (for RGB).
#   - Also, note that the original video image frames are 
#     made up of pixel values that are an INTEGER type 
#     and are between the range of 0-255 (again, for RGB).
TRAIN_VIDEO_PATH = "./train.mp4"
TRAIN_LABELS_PATH = "./train.txt"
HEIGHT=480
WIDTH=640
CHANNELS=3

TEST_VIDEO_PATH = "./test.mp4"
TEST_LABELS_PATH = "./test.txt"


# Generator function to retrieve frame & label from video.
#########################################################
# Note that original frame type is integer, RGB, 0-255
# so we convert it to tf.float32 and then divide it by 
# 255.0 to normalize pixel values to 0.0-1.0.
# -- this is useful later.
def gen(video_path, label_path):
    cap = cv2.VideoCapture(video_path.decode("utf-8"))

    with open(label_path.decode("utf-8")) as f:
        for line in f:
            _, frame = cap.read()
            yield (
                tf.convert_to_tensor(frame, dtype=tf.float32) / 255.0, 
                # tf.convert_to_tensor(frame, dtype=tf.uint8), 
                tf.convert_to_tensor(float(line), tf.float64)
            )

    cap.release()
    return


# Augmentation functions (per frame)
######################################
def to_greyscale(frame, label):
    frame = tf.image.rgb_to_grayscale(frame)
    return frame, label

def shrink_by_half_w_resize(frame, label):
    '''
    Shrink the frame, returning a new frame with
    half the original height and half the original width.

    In the case of the comma.ai training dataset, the 
    original (H=480, W=640) frames will be halved to 
    (H=240, W=320).
    '''
    height, width, _ = frame.shape
    frame = tf.image.resize(frame, [height//2, width//2], antialias=True)
    return frame, label

def shrink_w_resize_with_crop_or_pad(frame, label):
    '''
    Shrink the frame from size 480,640 
    to 240,320. Note that the returned frame
    is of type tf.uint8.
    '''
    frame = tf.image.resize_with_crop_or_pad(frame, 240, 320)
    return frame, label

def shrink_by_a_lil(frame, label):
    frame = tf.image.resize(frame, [255, 340], antialias=True)
    return frame, label

# Dataset helper functions.
#############################

# Gets the original dataset.
def get_original_dataset(video_path=TRAIN_VIDEO_PATH, labels_path=TRAIN_LABELS_PATH):
    ds = tf.data.Dataset.from_generator(
            gen, 
            # output_types=(tf.uint8, tf.float64),
            output_types=(tf.float32, tf.float64),
            output_shapes=([HEIGHT, WIDTH, CHANNELS], []),
            args=(video_path, labels_path)
        )

    return ds

# Applies augmentations to original dataset.
def augment_image_frames(ds, augs=[to_greyscale, shrink_by_a_lil]):
    for aug in augs:
        ds = ds.map(aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds

# Merges window_size number of sequential frames into one stacked 
# tensor of shape (H, W, window_size).
def merge_image_frames(ds, window_size):
    window_batches = ds.batch(window_size)
    return window_batches.map(
            _get_batch_to_window(window_size), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

def _get_batch_to_window(window_size):
    def _batch_to_window(frames_batch, labels_batch):
        window = tf.transpose(tf.squeeze(frames_batch), perm=[1, 2, 0])

        # NOTE: This must be set to the correct size.
        window = tf.ensure_shape(window, [255, 340, window_size])
        return window, labels_batch

    return _batch_to_window

### Old function version.
#####################################################
## NOTE: this uses the constants for a window size of four + Height & Width.
#
def _batch_to_window(frames_batch, labels_batch):
    window = tf.transpose(tf.squeeze(frames_batch), perm=[1, 2, 0])

    # NOTE: This must be set to the correct size.
    window = tf.ensure_shape(window, [255, 340, 4])
    return window, labels_batch

# Key dataset method ==> get the final pre-processed dataset.
#   - greyscale
#   - shrunk in half
#   - stacked sequential image frames (num of frames given by window_size)
#####################################################
def get_pre_processed_train_dataset(window_size=4):
    original_ds = get_original_dataset()
    aug_ds = augment_image_frames(original_ds)
    return merge_image_frames(aug_ds, window_size)

def get_pre_processed_test_dataset(window_size=4):
    original_ds = get_original_dataset(TEST_VIDEO_PATH, TEST_LABELS_PATH)
    aug_ds = augment_image_frames(original_ds)
    return merge_image_frames(aug_ds, window_size)

