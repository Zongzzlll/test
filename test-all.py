import tensorflow as tf
import numpy as np
import glob
import os.path
import cv2
import shutil


# 模型目录
CHECKPOINT_DIR = './runs/1576313317/checkpoints/'
INCEPTION_MODEL_FILE = 'inception/classify_image_graph_def.pb'
path = "./images_analysis_all" 
imagelist = os.listdir(path)
correct=0
filelist = os.listdir(path)  # 打开对应的文件夹
total_num = len(filelist)
# inception-v3模型参数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

# 测试数据
# file_path = './flower_photos/rose/rose_11.jpg'
# file_path = './flower_photos/roses/12240303_80d87f77a3_n.jpg'
# file_path = './data/flower_photos/dandelion/7355522_b66e5d3078_m.jpg'
# file_path = '/home/zstu913-server/flower_demo/test-photos/dandelion_14.jpg'
# file_path = './data/flower_photos/sunflowers/6953297_8576bf4ea3.jpg'
# file_path = './data/flower_photos/sunflowers/40410814_fba3837226_n.jpg'
# file_path = './data/flower_photos/tulips/11746367_d23a35b085_n.jpg'
y_test = ["Sunflower","Dandelion","Fritillary","Buttercup","Pansy","Crocus","Tigerlily","Snowdrop","Colts'Foot","LilyValley","Bluebell","Tulip","Daisy","Cowslip","Windflower","Daffodil","Iris"]

# 读取数据
#image_data = tf.gfile.GFile(file_path, 'rb').read()

# 评估
for imgname in imagelist:
    if(imgname.endswith(".jpg")):
       image_data = tf.gfile.FastGFile(os.path.join(path, imgname), 'rb').read()
       print(imgname)
       checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
       with tf.Graph().as_default() as graph:
           with tf.Session().as_default() as sess:
        # 读取训练好的inception-v3模型
             with tf.gfile.GFile(INCEPTION_MODEL_FILE, 'rb') as f:
                  graph_def = tf.GraphDef()
                  graph_def.ParseFromString(f.read())

        # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量
                  bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                      graph_def,
                       return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

        # 使用inception-v3处理图片获取特征向量
                  bottleneck_values = sess.run(bottleneck_tensor,
                                                {jpeg_data_tensor: image_data})
        # 将四维数组压缩成一维数组，由于全连接层输入时有batch的维度，所以用列表作为输入
                  bottleneck_values = [np.squeeze(bottleneck_values)]

        # 加载元图和变量
                  saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
                  saver.restore(sess, checkpoint_file)

        # 通过名字从图中获取输入占位符
                  input_x = graph.get_operation_by_name(
                       'BottleneckInputPlaceholder').outputs[0]

        # 我们想要评估的tensors
                  predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[
                       0]

        # 收集预测值
                  all_predictions = []
                  all_predictions = sess.run(predictions, {input_x: bottleneck_values})
                  a=all_predictions[0]
                  if(y_test[a]==imgname.split('_')[0]):
                       correct+=1
                  #print(sum(all_predictions == y_test))
print("Correct number is: " + str(correct))
print("Wrong number is: " + str(total_num-correct))
print("The accuracy rate of the test is: {:.2f}%".format(correct/total_num*100))

