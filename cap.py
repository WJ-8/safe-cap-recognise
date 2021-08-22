import numpy as np
import tensorflow as tf
import cv2
import os

from object_detection.utils import visualization_utils as vis_util


#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cv2.setUseOptimized(True)  # 加速cv

PATH_TO_CKPT = 'frozen_inference_graph.pb'  # 模型及标签地址
PATH_TO_LABELS = 'label_map.pbtxt'

NUM_CLASSES = 2  # 检测对象个数

camera_num = 0  # 要打开的摄像头编号，可能是0或1
width, height = 1280, 720  # 视频分辨率

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
mv = cv2.VideoCapture(camera_num)  # 打开摄像头

mv.set(9, width)  # 设置分辨率
mv.set(16, height)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while True:
            ret, img = mv.read()  # 读取视频帧
           # img = cv2.imread('1.jpg')
            rows = img.shape[0]
            cols = img.shape[1]
            inp = cv2.resize(img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

            # Run the model
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={'image_tensor:0': inp. reshape(1, inp.shape[0], inp.shape[1], 3)})
            # Visualization of the results of a detection.
            num_detections = int(out[0][0])
            num1 = 0  # 帽子计数
            num2 = 0  # 无帽子计数
            for i in range(num_detections):
                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                if score > 0.5:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    if classId == 1:
                        cl = (0, 255, 0)
                        num1 += 1
                    if classId == 2:
                        cl = (0, 0, 255)
                        num2 += 1
                    cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), cl, thickness=2)  # (B,G,R)

            font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
            num_1 = cv2.putText(img, 'cap:' + str(num1), (10, 25), font, 0.7, (0, 255, 0), 2)
            num_2 = cv2.putText(img, 'non:' + str(num2), (10, 50), font, 0.7, (0, 0, 255), 2)
            cv2.imshow("Helmet identification", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                break
mv.release()
cv2.destroyAllWindows()
