# encoding:utf-8
import tensorflow as tf

def get_accuracy(pred, label, label_mask, epsilon=1e-5):
    """
    计算模型预测的准确率指标
    功能：
        计算在给定掩码条件下, 预测值与真实标签值的匹配准确率
        用于评估分类模型的性能
    输入：
        pred: 张量, 模型的预测标签值
        label: 张量, 真实标签值
        label_mask: 张量, 标签掩码, 用于过滤需要参与计算的样本
        epsilon: 浮点数, 防止除零错误的小值, 默认为1e-5
    输出：
        张量, 计算得到的准确率值
    """
    # 计算预测正确的样本数量(考虑掩码)
    cor_sum = tf.reduce_sum(
        tf.cast(
            tf.equal(pred, label), tf.float32) * label_mask
    )

    # 计算平均准确率
    cor_avg = cor_sum / tf.maximum(tf.reduce_sum(label_mask), epsilon)

    # 处理掩码为0的情况, 返回1.0表示完美准确率
    pad_accuracy = tf.where(
        tf.reduce_sum(label_mask) > 0.0,
        cor_avg,
        1.0
    )
    return pad_accuracy

def get_precision(pred, label, epsilon=1e-5):
    """
    计算模型预测的精确率指标
    功能：
        计算预测为正类的样本中实际为正类的比例
        用于评估模型在正类预测上的准确性
    输入：
        pred: 张量, 模型的二分类预测结果(0或1)
        label: 张量, 真实标签(0或1)
        epsilon: 浮点数, 防止除零错误的小值, 默认为1e-5
    输出：
        张量, 计算得到的精确率值
    """
    # 计算真正例(TP)数量：预测为正且实际为正的样本数
    tp_num = tf.reduce_sum(
        pred * label
    )
    # 计算预测为正的样本总数(TP+FP)
    tp_fp_num = tf.maximum(tf.reduce_sum(pred), epsilon)
    # 处理无正例预测的情况, 返回1.0表示完美精确率
    pad_precision = tf.where(
        tf.reduce_sum(pred) > 0.0,
        tp_num / tp_fp_num,
        1.0
    )
    return pad_precision

def get_recall(pred, label, epsilon=1e-5):
    """
    计算模型预测的召回率指标
    功能：
        计算实际为正类的样本中被预测为正类的比例
        用于评估模型识别正类样本的能力
    输入：
        pred: 张量, 模型的二分类预测结果(0或1)
        label: 张量, 真实标签(0或1)
        epsilon: 浮点数, 防止除零错误的小值, 默认为1e-5
    输出：
        张量, 计算得到的召回率值
    """
    # 计算真正例(TP)数量：预测为正且实际为正的样本数
    tp_num = tf.reduce_sum(
        pred * label
    )
    # 计算实际为正的样本总数(TP+FN)
    tp_fn_num = tf.maximum(tf.reduce_sum(label), epsilon)
    # 处理无正例的情况, 返回1.0表示完美召回率
    pad_recall = tf.where(
        tf.reduce_sum(label) > 0.0,
        tp_num / tp_fn_num,
        1.0
    )
    return pad_recall

# In-batch AUC evaluation
def get_AUC(pred, label, label_mask=None):
    """
    计算批量内的AUC(Area Under the Curve)指标
    功能：
        计算ROC曲线下的面积, 用于评估二分类模型的性能
        支持带掩码的AUC计算
    输入：
        pred: 一维张量, 模型的预测概率值
        label: 一维张量, 真实标签(0或1)
        label_mask: 一维张量, 标签掩码, 用于过滤需要参与计算的样本, 默认为None
    输出：
        张量, 计算得到的AUC值
    """
    # 确保输入是一维张量
    assert (len(pred.shape) == 1)
    assert (len(label.shape) == 1)
    
    # 创建标签比较矩阵, 表示样本i的标签大于样本j的标签
    label_greater_mat = (tf.reshape(label, [-1, 1]) > label)
    
    # 处理掩码情况
    if label_mask is not None:
        assert (len(label_mask.shape) == 1)
        label_mask = (label_mask > 0.0)
        label_greater_mat &= label_mask  # 过滤掉掩码为0的样本
        label_greater_mat &= tf.reshape(label_mask, [-1, 1])  # 过滤掉掩码为0的样本
    
    # 重塑预测值为列向量
    pred_col = tf.reshape(pred, [-1, 1])
    # 创建预测值比较矩阵, 表示样本i的预测值大于样本j的预测值
    pred_greater_mat = (pred_col > pred)
    # 创建预测值相等矩阵, 表示样本i的预测值等于样本j的预测值
    pred_equal_mat = tf.equal(pred_col, pred)
    
    # 计算总有效样本对数量
    total_pairs = tf.reduce_sum(tf.cast(label_greater_mat, tf.float32))
    
    # 计算满足条件的样本对数量(预测值排序与标签排序一致)
    # 对于预测值相等的情况, 权重为0.5
    greater_sum = tf.reduce_sum(
        tf.cast(pred_greater_mat & label_greater_mat, tf.float32)) + tf.reduce_sum(
        tf.cast(pred_equal_mat & label_greater_mat, tf.float32)) / 2.0
    
    # 计算AUC值, 处理无效情况
    return tf.where(total_pairs > 0.5, greater_sum / tf.maximum(total_pairs, 1.0), 1.0)

def wrap_metrics(metrics_dict):
    """
    包装评估指标字典, 用于TensorFlow的评估指标操作
    功能：
        将指标字典中的每个指标值计算平均值, 便于模型评估和日志记录
    输入：
        metrics_dict: 字典, 包含各种评估指标的键值对
    输出：
        字典, 包含每个指标的平均值
    """
    update_dict = dict()
    # 对每个指标计算平均值
    for k, v in metrics_dict.items():
        update_dict[k] = tf.reduce_mean(v)
        # update_dict[k] = (v, tf.no_op())
    return update_dict
