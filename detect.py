# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
from flask import jsonify

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
from flask import Flask, request
from werkzeug.utils import secure_filename
import numpy as np
app = Flask(__name__)
@app.route('/run', methods=['POST'])

def run():
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  #cap = cv2.VideoCapture("rtsp://office:office123@58.65.164.43:554/cam/realmonitor?channel=4&subtype=0")#camera_id)
  #cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model#model
  base_options = core.BaseOptions(
      file_name="efficientdet_lite0.tflite", use_coral=False, num_threads=4)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.00056)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  #while cap.isOpened():
    #success, image = cap.read()
  #image_path = request.files['image']
  image = request.files['image']
  image_data = image.read()
  # convert image to numpy array
  image_array = cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_COLOR)
  # apply grayscale filter
  image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
  # Convert the image to a numpy array
  #image_array = np.frombuffer(image.read(), np.uint8)
  # Read the image using OpenCV
  #image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
  #image=cv2.imread(image)#image_path)#'./test_data/test.jpg'
  """if not success:
    sys.exit(
        'ERROR: Unable to read from webcam. Please verify your webcam settings.'
    )"""

  counter += 1
  image = cv2.flip(image, 1)

  # Convert the image from BGR to RGB as required by the TFLite model.
  rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Create a TensorImage object from the RGB image.
  input_tensor = vision.TensorImage.create_from_array(rgb_image)

  # Run object detection estimation using the model.
  detection_result = detector.detect(input_tensor)
  #print(detection_result.detections)
  mylist=[]
  count=0
  for person in detection_result.detections:
    count=count+1
    #print(count)
    #print(person.categories[0].category_name)
    if person.categories[0].category_name=='person':
      mylist.append(count)
  print(len(mylist))
  

  #print(detection_result.detections)
  """if detection_result.detections[0].categories[0].category_name=='person':
    print(detection_result)
    #print("person")
    pass
  else:
    continue"""

  # Draw keypoints and edges on input image
  image = utils.visualize(image, detection_result)

  # Calculate the FPS
  if counter % fps_avg_frame_count == 0:
    end_time = time.time()
    fps = fps_avg_frame_count / (end_time - start_time)
    start_time = time.time()

  # Show the FPS
  fps_text = 'FPS = {:.1f}'.format(fps)
  text_location = (left_margin, row_size)
  cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
              font_size, text_color, font_thickness)

  # Stop the program if the ESC key is pressed.
  #if cv2.waitKey(1) == 27:
    # break#
  cv2.imwrite('object_detector.jpg', image)
  return jsonify({"peoplecount":len(mylist)})

  #cap.release()
  #cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  """parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()"""

  app.run(debug=True)


if __name__ == '__main__':
  main()
