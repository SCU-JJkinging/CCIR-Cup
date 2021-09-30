由于本地训练和线上训练存在一定偏差，所以提供了两种方式供复现
* 线上训练 + 线上推理 — run.sh
* 本地训练 + 线上推理 — run_infer.sh

准备工作：<br>
先从ccir/data/user_data/user_data.txt中的百度云网盘链接下载user_data.zip，然后将其解压并替换掉user_data目录；
然后从ccir/image/ccir-image.txt的百度云网盘链接下载镜像文件ccir-image.tar，并将其放在ccir/image目录下

1.线上训练 + 线上推理

  我的项目名字叫 ccir 

  这种方式是在线上重新训练模型然后推理复现，线上一共需要训练三个模型，在 T4 GPU 上一共大约需要耗时7个小时<br>
  镜像挂载运行方式：

  ```
  nvidia-docker run -v /home/seatrend/jinxiang/ccir/:/ccir ccir-image sh /ccir/image/run.sh
  ```

  其中：

  ```
  /home/seatrend/jinxiang/ccir/
  ```

  是我本地项目 ccir 所在的绝对路径，线上复现时需要替换成ccir在线上的绝对路径名，然后把它挂载到镜像的 /ccir 目录下，镜像名称是 ccir-image

2.本地训练 + 线上推理

   如果需要直接复现线下训练结果，则：

   ```
   nvidia-docker run -v /home/seatrend/jinxiang/ccir/:/ccir ccir-image sh /ccir/image/run_infer.sh
   ```
其中：

  ```
  /home/seatrend/jinxiang/ccir/
  ```

  是我本地项目 ccir 所在的绝对路径，线上复现时需要替换成ccir在线上的绝对路径名，然后把它挂载到镜像的 /ccir 目录下，镜像名称是 ccir-image

   线下训练好的模型共三个，分别保存在： 

   ```
   ccir/data/user_data/output_model/JointBert/trained_model 
   ```

   ```
   ccir/data/user_data/output_model/InteractModel_1/trained_model 
   ```

   ```
   ccir/data/user_data/output_model/InteractModel_3/trained_model 
   ```

   

