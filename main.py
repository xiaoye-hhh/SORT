import os
import numpy as np
import glob
import os.path as osp
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from skimage import io
import matplotlib.patches as patches
from utils.deal_config import read_config


def calc_iou(boxes1: np.ndarray, boxes2: np.ndarray):
    """ 计算IOU, 抓住左上角和右下角     

    Args:
        boxes1 (np.ndarray): [[x1, y1, x2, y2], [x1, y1, x2, y2]...]
        boxes2 (np.ndarray): [[x1, y1, x2, y2], [x1, y1, x2, y2]...]
    """
    boxes1 = np.expand_dims(boxes1, 1)
    boxes2 = np.expand_dims(boxes2, 0)

    max_x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    max_y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    min_x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    min_y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    w = np.maximum(0, min_x2 - max_x1)
    h = np.maximum(0, min_y2 - max_y1)
    s = w * h
    return s / ((boxes1[..., 2]-boxes1[..., 0]) * (boxes1[..., 3]-boxes1[..., 1]) + 
                (boxes2[..., 2]-boxes2[..., 0])*(boxes2[..., 3]-boxes2[..., 1]) - s)


def linear_assignment(cost_matirx: np.ndarray):
    """ 匈牙利算法: 将返回的结果打包成 [[x1, y1], [x1, y1]...] """
    x, y = linear_sum_assignment(cost_matirx)
    return np.array(list(zip(x, y)))


def associate_preditcts_to_detections(predicts: np.ndarray, detections: np.ndarray, iou_threshold=0.1):
    """ 匹配检测结果和追踪结果

    Args:
        trackers (np.ndarray): 追踪结果 [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        detections (np.ndarray): 检测结果 [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        iou_threshold (float, optional): iou的最低阈值. Defaults to 0.3.
    
    返回: matches: [[index1, index2], [index1, index2]...]
         unmatched_detections: [index1, index2 ...]
         unmatched_trackers: [index1, index2...]
    """
    # 先根据IOU值去匹配，再挑选出IOU值符合的匹配
    iou_matrix = calc_iou(predicts, detections)
    candidate_matched_indices = linear_assignment(-iou_matrix)
    
    matched_indices = []
    for inidice in candidate_matched_indices:
        if iou_matrix[inidice[0], inidice[1]] > iou_threshold:
            matched_indices.append(inidice)
    matched_indices = np.array(matched_indices) if len(matched_indices) else np.empty((0, 2))

    # 提取未匹配的
    unmatched_detections = []
    for i in range(len(detections)):
        if i not in matched_indices[:, 1]:
            unmatched_detections.append(i)

    unmatched_trackers = []
    for i in range(len(predicts)):
        if i not in matched_indices[:, 0]:
            unmatched_trackers.append(i)

    return matched_indices, unmatched_trackers, unmatched_detections


def convert_box_to_center(box: np.ndarray):
    """ 转换坐标表示: [x1, y1, x2, y2] ==> [x, y, s, r]
        x,y: 是中心坐标; s是面积; r是宽高比例

    Args:
        box (np.ndarray): [x1, y1, x2, y2]表示的矩形框
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = box[0] + w / 2
    y = box[1] + h / 2
    s = w * h
    r = w / h
    return np.array([x, y, s, r])


def convert_center_to_box(c: np.ndarray):
    """ 转换坐标表示: [x, y, s, r] ==> [x1, y1, x2, y2] 
        如果有score, 则变成 [x1, y1, x2, y2, score] 
    """
    w = np.sqrt(c[2] * c[3])
    h = c[2] / w

    return np.array([c[0] - w/2, c[1] - h/2, c[0] + w/2, c[1] + h/2])


class KF:
    count = 0
    def __init__(self, state) -> None:
        """
            状态: [x, y, s, r, x', y', s']
            观测量: [x, y, s, r]
        """
        
        KF.count += 1
        self.id = KF.count
        self.age = 0

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        
        self.kf.P[4:, 4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.R[2:, 2:] *= 1
        self.kf.Q[-1, -1] *= 0.1
        self.kf.Q[4:, 4:] *= 0.1

        self.kf.x[:4] = state.reshape(-1, 1)

 
    def predict(self):
        self.age += 1
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        return self.kf.x[:4]


    def update(self, detection):
        self.kf.update(detection.reshape(-1, 1))


    def get_state(self):
        return self.kf.x[:4]


class Manager:
    def __init__(self, min_age, iou_threshold) -> None:
        self.min_age = min_age
        self.iou_threshold = iou_threshold
        self.kfs = []

    def update(self, detections):
        # =====================================================
        # 1.根据kf求出预测值
        # 2.删除不合理的kf
        # 3.删除不合理的预测值
        # ===================================================== 
        to_del = []
        predict_pos = np.zeros((len(self.kfs), 4), np.float32)
        for i, kf in enumerate(self.kfs):
            predict_pos[i] = convert_center_to_box(kf.predict()[:, 0])
            if np.any(np.isnan(predict_pos[i])):
                to_del.append(i)
        
        for i in reversed(to_del):
            self.kfs.pop(i)

        predict_pos = np.ma.compress_rows(np.ma.masked_invalid(predict_pos))

        # =====================================================
        # 1.做匈牙利匹配
        # 2.对匹配的kf做一个更新
        # 3.删除为匹配的kf
        # 4.对为匹配的对象创建一个kf
        # =====================================================
        matched_indices, unmatched_kfs, unmatched_detections = associate_preditcts_to_detections(predict_pos, detections, self.iou_threshold)
        for i, j in matched_indices:
            self.kfs[i].update(convert_box_to_center(detections[j]))
        
        for i in reversed(unmatched_kfs):
            self.kfs.pop(i) 
        
        for i in unmatched_detections:
            self.kfs.append(KF(convert_box_to_center(detections[i])))

        # 构造返回值
        ret = []
        for kf in self.kfs:
            if kf.age >= self.min_age:
                tmp = kf.get_state().reshape(-1)
                tmp = convert_center_to_box(tmp)
                ret.append(np.concatenate([tmp, np.array([kf.id])]))

        return ret


if __name__ == '__main__':
    config = read_config()

    if config.display:
        colors = np.random.rand(32, 3)
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    for path in glob.glob(config.train_parttern):
        name = path.split('/')[-3]
        print(f"正在处理: {name}")
        manager = Manager(config.min_age, config.iou_threshold)

        # 加载数据，并转换 [frame, _, x1, y1, w, h, _, _, _], ==> [frame, _, x1, y1, x2, y2, _, _, _, _]
        data = np.loadtxt(path, delimiter=',')
        data[:, 4:6] += data[:, 2:4] 

        with open(osp.join(config.out_dir, f'{name}.txt'), 'w') as out_file:
            for frame in range(1, int(data[:, 0].max())):
                # 取一帧的信息:[[x1, y1, x2, y2], [x1, y1, x2, y2]...]
                detections = data[data[:, 0]==frame, 2:6]

                # 更新结果
                boxes = manager.update(detections)
                
                if config.display:
                    fn = os.path.join(config.test_dir, name, 'img1', f'{frame:06d}.jpg')
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(name)

                # 输出到文件，格式: [[frame, x, y, w, h], [frame, x, y, w, h]...]
                for box in boxes:
                    print(f'{frame:d},{box[0]:.2f},{box[1]:.2f},{box[2]-box[0]:.2f},{box[3]-box[1]:.2f}', file=out_file)
                    if config.display:
                        box = box.astype(np.int32)
                        ax1.add_patch(patches.Rectangle(
                            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            fill=False, lw=3, ec= colors[box[4]%32,:]))


                if config.display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()