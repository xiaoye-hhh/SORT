# 对SORT的复现
[数据集链接](https://motchallenge.net/data/MOT15.zip)

效果看起来不是很好, 准备尝试deepsort
# 快速开始
下载代码: `git clone git@github.com:xiaoye-hhh/SORT.git`
下载数据集: `wget https://motchallenge.net/data/MOT15.zip`
解压: `unzip MOT15.zip`
改名: `mv MOT15.zip data`
运行: `python main.py`

# 实现了一个简单的deep_sort
[链接](https://github.com/xiaoye-hhh/simple_deepsort)

一些坑:
 - 建议使用:filterpy的卡尔曼滤波,而不是cv2的
 - 注意区分矩形框的表示格式: 
   - 数据集格式: [frame, _, x1, y1, w, h, _, _, _]
   - 计算IOU格式: [x1, y1, x2, y2]
   - 模型里的格式: [x, y, s, r]

@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}
