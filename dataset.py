#!/Users/dguillen/opt/anaconda3/bin/python3

import cv2
import itertools
import tensorflow as tf

def gen(video_path, label_path):
    cap = cv2.VideoCapture(video_path.decode("utf-8"))

    with open(label_path.decode("utf-8")) as f:
        for line in f:
            _, frame = cap.read()
            yield (
                tf.convert_to_tensor(frame), 
                tf.convert_to_tensor(float(line), tf.float64)
            )

    cap.release()
    return

'''
for frame, speed in gen("./train.mp4", "./train.txt"):
    print(f"Frame type: {frame.dtype}, Speed type: {speed.dtype}")
    print(speed)
    break
'''

class VideoDataGenerator:
    def __iter__(self, video_path, label_path):
        self.video_path = video_path
        self.label_path = label_path
        return self
    
    def __next__(self):
        gen(self.video_path, self.label_path)

ds = tf.data.Dataset.from_generator(
        gen, 
        output_types=(tf.uint8, tf.float64),
        output_shapes=([480, 640, 3], []),
        args=("./train.mp4", "./train.txt")
    )

for frame, speed in ds:
    print(frame.shape)
    print(speed)
    break
