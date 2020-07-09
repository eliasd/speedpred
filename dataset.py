#!/Users/dguillen/opt/anaconda3/bin/python3

import cv2
import itertools
import tensorflow as tf


# Generator function to retrieve frame & label from video.
#########################################################
# Note that frame type is integer, RGB, 0-255
# so we convert to tf.float32 + divide by 255.0
# to normalize pixel values to 0.0-1.0
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

# Construct dataset from original data.
########################################
ds = tf.data.Dataset.from_generator(
        gen, 
        # output_types=(tf.uint8, tf.float64),
        output_types=(tf.float32, tf.float64),
        output_shapes=([480, 640, 3], []),
        args=("./train.mp4", "./train.txt")
    )

# Visualize original frames.
'''
print()
for frame, speed in ds:
    print('Original frame type: {}'.format(frame.dtype))
    print("Original shape: {}".format(frame.shape))
    print()
    cv2.imshow("frame", frame.numpy())
    cv2.waitKey()
    break
'''

# Augmentation functions (per frame)
######################################
def to_greyscale(frame, label):
    frame = tf.image.rgb_to_grayscale(frame)
    return frame, label

def shrink_w_resize(frame, label):
    '''
    Shrink the frame from size H=480,W=640
    to H=240,W=320. Note that the returned frame
    is of type tf.float32, which can cause display
    headaches unless casted to tf.uint8 or 
    divided by 255.
    '''
    frame = tf.image.resize(frame, [240, 320], antialias=True)
    return frame, label

def shrink_w_resize_with_crop_or_pad(frame, label):
    '''
    Shrink the frame from size 480,640 
    to 240,320. Note that the returned frame
    is of type tf.uint8.
    '''
    frame = tf.image.resize_with_crop_or_pad(frame, 240, 320)
    return frame, label

# Apply frame-specific augmentations.
#####################################
#   - convert rgb image to grey-scale.
#   - resize image from () to ().
augmentations = [to_greyscale, shrink_w_resize]
for aug in augmentations:
    ds = ds.map(aug)

# Visualize augmented frames.
'''
for frame, speed in ds:
    print('Augmented frame type: {}'.format(frame.dtype))
    print("Augmented frame shape: {}".format(frame.shape))
    print()
    cv2.imshow("frame", frame.numpy())
    cv2.waitKey()
    break
'''

# TODO: Create windows from sequential frames dataset.
#       To do this, batch the dataset by the window size
#       and then .map(func) where func reshapes or transposes
#       the frames such that the final shape is (H, W, WINDOW_SIZE).
# four_window_batches = ds.batch(4)

# TODO: Create function that creates dataset, applies augmentations,
#       and groups the frames together into windows, with the 
#       WINDOW_SIZE being passed in as a custom parameter.


