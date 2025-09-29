# encoding:utf-8

import tensorflow as tf
import math


def get_vars_in_scope(name_scope="model_bias"):
    """
    获取指定作用域内的所有可训练变量
    功能：
        返回TensorFlow图中指定名称作用域下的所有可训练变量, 默认获取model_bias作用域的变量
    输入：
        name_scope: 字符串, 变量作用域名称, 默认为"model_bias"
    输出：
        列表, 包含指定作用域内的所有可训练变量
    """
    # print([v.name for v in tf.compat.v1.trainable_variables(
    #     scope=name_scope
    # )])
    return tf.compat.v1.trainable_variables(
        scope=name_scope
    )

def get_vars_not_in_scope(name_scope="model_bias"):
    """
    获取不在指定作用域内的所有可训练变量
    功能：
        返回TensorFlow图中不在指定名称作用域下的所有可训练变量
    输入：
        name_scope: 字符串, 变量作用域名称, 默认为"model_bias"
    输出：
        列表, 包含不在指定作用域内的所有可训练变量
    """
    # print([v.name for v in tf.compat.v1.trainable_variables() if not v.name.startswith(name_scope)])
    return [v for v in tf.compat.v1.trainable_variables() if not v.name.startswith(name_scope)]

def huber_loss(labels, pred, logging_hook_di, delta=1.0):
    """
    计算Huber损失函数
    功能：
        计算预测值与真实值之间的Huber损失, 该损失对异常值具有鲁棒性
        当预测误差较小时(小于delta), 表现为平方损失；当误差较大时, 表现为线性损失
        同时将相关变量记录到日志钩子字典中, 便于调试和监控
    输入：
        labels: 张量, 真实值标签
        pred: 张量, 模型预测值
        logging_hook_di: 字典, 用于记录调试信息的日志钩子字典
        delta: 浮点数, Huber损失的阈值参数, 默认为1.0
    输出：
        张量, 计算得到的Huber损失值
    """
    # 将真实标签和预测值记录到日志字典中
    logging_hook_di['huber_labels'] = labels
    logging_hook_di['huber_pred'] = pred
    
    # 计算预测值与真实值之间的绝对误差
    residual = tf.abs(pred - labels)
    
    # 判断误差是否小于delta阈值, 用于决定使用哪种损失计算方式
    condition = tf.less(residual, delta)
    
    # 当误差较小时使用平方损失(二次损失)
    small_res = 0.5 * tf.square(residual)
    
    # 当误差较大时使用线性损失, 避免异常值导致的梯度爆炸
    large_res = delta * residual - 0.5 * tf.square(delta)
    
    # 将计算的线性损失记录到日志字典中
    logging_hook_di['huber_loss'] = large_res
    
    # 根据误差大小选择对应的损失计算方式, 返回最终的Huber损失
    return tf.where(condition, small_res, large_res)

def weighted_cross_entropy_with_logits(labels, pred):
    """
    计算带权重的交叉熵损失(使用logits输入)
    功能：
        计算基于logits的带权重交叉熵损失, 对正样本赋予不同权重, 适用于类别不平衡问题
    输入：
        labels: 张量, 真实值标签
        pred: 张量, 模型预测的logits值
    输出：
        张量, 计算得到的带权重交叉熵损失
    """
    # 将标签二值化：大于0的为1, 否则为0
    binary_label = tf.where(
        labels > 0.0, tf.ones_like(labels), tf.zeros_like(labels)
    )
    # 计算带权重的交叉熵损失, 正样本权重为1+labels
    wce_loss = tf.nn.weighted_cross_entropy_with_logits(
        labels=tf.reshape(binary_label, [-1]),
        logits=pred,
        pos_weight=tf.reshape(1 + labels, [-1])
    )
    return wce_loss

# label分桶
def label_to_buckets(labels):
    """
    将标签值映射到对应的桶索引
    功能：
        对连续型标签值进行分桶处理, 将其转换为离散的桶索引
        用于将回归问题转换为分类问题, 实现知识蒸馏
    输入：
        labels: 张量, 原始连续型标签值
    输出：
        张量, 分桶后的标签索引, 形状为(bs, 1)
    """
    # 将标签展平为一维张量
    labels_flat = tf.reshape(labels, [-1])
    
    # 定义分桶的边界值, 将标签分成6个桶
    boundaries =  [1e-6, 1.0, 5.0, 50.0, 100.0]
    
    # 使用TensorFlow的Bucketize操作将标签值分配到对应的桶中
    bucket_labels = tf.raw_ops.Bucketize(input=labels_flat, boundaries=boundaries)
    
    # 返回分桶后的标签索引
    return bucket_labels  # (bs, 1)

def distill_softmax(labels, pred, logging_hook_di):
    """
    使用分桶标签计算蒸馏软max交叉熵损失
    功能：
        将连续型标签通过分桶转换为离散类别, 然后计算交叉熵损失
        同时将相关变量记录到日志钩子字典中, 便于调试和监控
    输入：
        labels: 张量, 原始连续型标签值
        pred: 张量, 模型预测的logits值
        logging_hook_di: 字典, 用于记录调试信息的日志钩子字典
    输出：
        张量, 计算得到的交叉熵损失值
    """
    # 将原始标签记录到日志字典中
    logging_hook_di['labels'] = labels
    
    # 对标签进行分桶处理
    bucket_labels = label_to_buckets(labels)
    
    # 将分桶后的标签记录到日志字典中
    logging_hook_di['bucket_labels'] = bucket_labels

    # 计算稀疏软max交叉熵损失
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels = bucket_labels,
                                                          logits = pred )

def weighted_cross_entropy(labels, pred, weight, logging_hook_di):
    """
    计算自定义权重的交叉熵损失
    功能：
        根据提供的权重计算交叉熵损失, 并将中间结果记录到日志字典中
    输入：
        labels: 张量, 真实值标签
        pred: 张量, 模型预测的概率值
        weight: 张量, 样本权重
        logging_hook_di: 字典, 用于记录调试信息
    输出：
        张量, 计算得到的带权重交叉熵损失
    """
    # 记录真实标签和预测分数到日志字典
    logging_hook_di['[lzx_debug] task_label'] = labels
    logging_hook_di['[lzx_debug] task_score'] = pred

    # 将标签二值化：大于0的为1, 否则为0
    binary_label = tf.where(
        labels > 0.0, tf.ones_like(labels), tf.zeros_like(labels)
    )

    # logging_hook_di['[lzx_debug] binary_label'] = binary_label
    
    # 计算带权重的交叉熵损失
    wce_loss = -binary_label * tf.log(pred) * weight - (1 - binary_label) * tf.log(1 - pred)

    # 记录计算得到的损失值
    logging_hook_di['[lzx_debug] wce_loss'] = wce_loss
    
    return wce_loss

def focal_loss(labels, pred):
    """
    计算Focal Loss损失函数
    功能：
        实现用于解决类别不平衡问题的Focal Loss损失函数
        对简单样本的损失进行衰减, 使模型更关注难分类的样本
    输入：
        labels: 张量, 真实标签值(0或1)
        pred: 张量, 模型预测的概率值(在0到1之间)
    输出：
        张量, 计算得到的Focal Loss值
    """
    # Focal Loss公式(alpha为正负样本权重, gamma为难度系数, gamma>0)
    # alpha=0.25表示正样本权重较小, 用于处理类别不平衡问题
    alpha = 0.25
    # gamma=2.0控制难易样本的区分度, 值越大对简单样本的惩罚越小
    gamma = 2.0
    
    # 计算Focal Loss：包括正样本部分和负样本部分
    # 正样本部分：-alpha * labels * log(pred) * (1-pred)^gamma
    # 负样本部分：-(1-alpha) * (1-labels) * log(1-pred) * pred^gamma
    focal_loss = -alpha * labels * tf.log(pred) * (1 - pred)**gamma \
             - (1 - alpha) * (1 - labels) * tf.log(1 - pred) * pred**gamma
    
    # 返回计算得到的Focal Loss值
    return focal_loss

def cross_entropy(labels, pred, logging_hook_di):
    """
    计算标准交叉熵损失
    功能：
        计算标准的二分类交叉熵损失, 包含对预测概率的裁剪以避免数值不稳定
    输入：
        labels: 张量, 真实值标签
        pred: 张量, 模型预测的概率值
        logging_hook_di: 字典, 用于记录调试信息
    输出：
        张量, 计算得到的交叉熵损失
    """
    # 记录真实标签和预测分数到日志字典
    logging_hook_di['[lzx_debug] task_label'] = labels
    logging_hook_di['[lzx_debug] task_score'] = pred
    
    # # 将标签二值化：大于0的为1, 否则为0
    # binary_label = tf.where(
    #     labels > 0.0, tf.ones_like(labels), tf.zeros_like(labels)
    # )
    # # 记录二值化后的标签
    # logging_hook_di['[lzx_debug] binary_label'] = binary_label
    # # 对预测概率进行裁剪, 避免数值不稳定
    # clipped_pred = tf.clip_by_value(pred, clip_value_min=0.000001, clip_value_max=0.999999)
    # # 记录裁剪后的预测概率
    # logging_hook_di['[lzx_debug] clipped_pred'] = clipped_pred

    # # 计算交叉熵损失
    # ce_loss = -binary_label * tf.log(clipped_pred) - (1 - binary_label) * tf.log(1 - clipped_pred)
    ce_loss = -labels * tf.math.log(pred) - (1 - labels) * tf.math.log(1 - pred)
    # ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=tf.reshape(binary_label, [-1]),
    #     logits=tf.reshape(pred, [-1])
    # )

    # 记录计算得到的损失值
    logging_hook_di['[lzx_debug] ce_loss'] = ce_loss

    return ce_loss

def binarized_reward(reward, cost_level_mask, key_to_mask, threshold=None):
    """
    对奖励值进行二值化处理
    功能：
        根据不同成本级别的阈值, 将连续奖励值转换为二值化奖励
        高成本和中等成本级别的样本使用不同的阈值判断
    输入：
        reward: 张量, 原始奖励值
        cost_level_mask: 张量, 样本的成本级别掩码
        key_to_mask: 字典, 包含不同成本级别的掩码值
        threshold: 列表或None, 包含中等和高成本级别的阈值, 默认为[11.2, 16.5]
    输出：
        张量, 二值化后的奖励值
    """
    # 获取高成本和中等成本级别的掩码值
    high_mask = key_to_mask['high']
    medium_mask = key_to_mask['medium']
    # 设置默认阈值
    if threshold is None:
        threshold = [11.2, 16.5]
    # 验证阈值参数的有效性
    assert len(threshold) == 2, "threshold: {} len != 2".format(threshold)
    assert threshold[0] <= threshold[1], "threshold: {} must le threshold: {} ".format(
        threshold[0], threshold[1]
    )
    # 计算中等成本级别样本的二值化奖励
    medium_reward = tf.cast(tf.math.logical_and(
        tf.equal(cost_level_mask, medium_mask), tf.greater_equal(reward, threshold[0])
    ), tf.float32)
    # 计算高成本级别样本的二值化奖励
    high_reward = tf.cast(tf.math.logical_and(
        tf.equal(cost_level_mask, high_mask), tf.greater_equal(reward, threshold[1])
    ), tf.float32)
    # 合并两种成本级别的二值化奖励
    merge_reward = high_reward + medium_reward
    # 记录中等成本级别二值化奖励的直方图
    tf.summary.histogram(
        "medium_reward", tf.boolean_mask(
            merge_reward,
            tf.equal(cost_level_mask, medium_mask)
        )
    )
    # 记录高成本级别二值化奖励的直方图
    tf.summary.histogram(
        "high_reward", tf.boolean_mask(
            merge_reward,
            tf.equal(cost_level_mask, high_mask)
        )
    )
    return merge_reward

def reward_shaping(reward, cost_level_mask, bill_ratio, key_to_mask):
    """
    对奖励值进行塑形处理
    功能：
        根据成本级别和计费比调整奖励值, 对不同类型的样本(如空跑、低估、高估等)应用不同的奖励系数
    输入：
        reward: 张量, 原始奖励值
        cost_level_mask: 张量, 每个样本映射到不同的成本级别(high, medium)
        bill_ratio: 张量, 计费比, 用于判断预测是否准确：
            (0.0, 0.8]: 低估
            0.8-1.2: 正常
            >1.2 高估
            0.0: 空跑
        key_to_mask: 字典, 包含不同成本级别的掩码值
    输出：
        张量, 塑形后的奖励值
    """
    # 获取高成本和中等成本级别的掩码值
    high_mask = key_to_mask['high']
    medium_mask = key_to_mask['medium']
    
    # 定义不同条件掩码
    # 中等成本级别且空跑
    is_medium_cost_level_and_empty_run = tf.math.logical_and(
        tf.equal(cost_level_mask, medium_mask),
        tf.equal(bill_ratio, 0.0)
    )
    # 中等成本级别且低估
    is_medium_cost_level_and_low_bill = tf.math.logical_and(
        tf.equal(cost_level_mask, medium_mask),
        tf.math.logical_and(
            tf.greater(bill_ratio, 0.0),
            tf.less_equal(bill_ratio, 0.8)
        )
    )
    # 中等成本级别且高估
    is_medium_cost_level_and_high_bill = tf.math.logical_and(
        tf.equal(cost_level_mask, medium_mask),
        tf.greater(bill_ratio, 1.3)
    )
    # 高成本级别且空跑
    is_high_cost_level_and_empty_run = tf.math.logical_and(
        tf.equal(cost_level_mask, high_mask),
        tf.equal(bill_ratio, 0.0)
    )
    # 高成本级别且低估
    is_high_cost_level_and_low_bill = tf.math.logical_and(
        tf.equal(cost_level_mask, high_mask),
        tf.math.logical_and(
            tf.greater(bill_ratio, 0.0),
            tf.less_equal(bill_ratio, 0.8)
        )
    )
    # 高成本级别且高估
    is_high_cost_level_and_high_bill = tf.math.logical_and(
        tf.equal(cost_level_mask, high_mask),
        tf.greater(bill_ratio, 1.3)
    )
    
    # 应用不同的奖励系数
    # 高成本级别空跑样本惩罚(0.1倍)
    reward = tf.where(
        is_high_cost_level_and_empty_run,
        reward * 0.1,
        reward
    )
    # 中等成本级别空跑样本惩罚(0.1倍)
    reward = tf.where(
        is_medium_cost_level_and_empty_run,
        reward * 0.1,
        reward
    )
    # 高成本级别高估样本轻微惩罚(0.8倍)
    reward = tf.where(
        is_high_cost_level_and_high_bill,
        reward * 0.8,
        reward
    )
    # 中等成本级别高估样本轻微惩罚(0.8倍)
    reward = tf.where(
        is_medium_cost_level_and_high_bill,
        reward * 0.8,
        reward
    )
    
    return reward

def reward_shaping_v2(reward, cost_level_mask, bill_ratio, key_to_mask):
    """
    奖励塑形函数的第二个版本
    
    功能：
        扩展了原始reward_shaping函数, 增加了对对照组和实验组的区分处理
        根据成本级别、实验分组和计费比调整奖励值
    
    输入：
        reward: 张量, 原始奖励值
        cost_level_mask: 张量, 每个样本映射到不同的成本级别
        bill_ratio: 张量, 计费比, 用于判断预测是否准确
        key_to_mask: 字典, 包含不同成本级别和实验分组的掩码值
    
    输出：
        张量, 塑形后的奖励值
    """
    # 获取不同成本级别和实验分组的掩码值
    high_control_mask = key_to_mask['high_and_control']
    high_treat_mask = key_to_mask['high_and_treat']
    medium_control_mask = key_to_mask['medium_and_control']
    medium_treat_mask = key_to_mask['medium_and_treat']
    
    # 定义不同条件掩码
    # 中等成本级别(对照组或实验组)且空跑
    is_medium_cost_level_and_empty_run = tf.math.logical_and(
        tf.math.logical_or(
            tf.equal(cost_level_mask, medium_control_mask),
            tf.equal(cost_level_mask, medium_treat_mask)
        ),
        tf.equal(bill_ratio, 0.0)
    )
    # 中等成本级别且低估
    is_medium_cost_level_and_low_bill = tf.math.logical_and(
        tf.math.logical_or(
            tf.equal(cost_level_mask, medium_control_mask),
            tf.equal(cost_level_mask, medium_treat_mask)
        ),
        tf.math.logical_and(
            tf.greater(bill_ratio, 0.0),
            tf.less_equal(bill_ratio, 0.8)
        )
    )
    # 中等成本级别且高估
    is_medium_cost_level_and_high_bill = tf.math.logical_and(
        tf.math.logical_or(
            tf.equal(cost_level_mask, medium_control_mask),
            tf.equal(cost_level_mask, medium_treat_mask)
        ),
        tf.greater(bill_ratio, 1.3)
    )
    # 高成本级别且空跑(注意这里使用了中等成本级别的掩码, 可能是代码错误)
    is_high_cost_level_and_empty_run = tf.math.logical_and(
        tf.math.logical_or(
            tf.equal(cost_level_mask, medium_control_mask),
            tf.equal(cost_level_mask, medium_treat_mask)
        ),
        tf.equal(bill_ratio, 0.0)
    )
    # 高成本级别(对照组或实验组)且低估
    is_high_cost_level_and_low_bill = tf.math.logical_and(
        tf.math.logical_or(
            tf.equal(cost_level_mask, high_control_mask),
            tf.equal(cost_level_mask, high_treat_mask)
        ),
        tf.math.logical_and(
            tf.greater(bill_ratio, 0.0),
            tf.less_equal(bill_ratio, 0.8)
        )
    )
    # 高成本级别(对照组或实验组)且高估
    is_high_cost_level_and_high_bill = tf.math.logical_and(
        tf.math.logical_or(
            tf.equal(cost_level_mask, high_control_mask),
            tf.equal(cost_level_mask, high_treat_mask)
        ),
        tf.greater(bill_ratio, 1.3)
    )
    
    # 应用不同的奖励系数
    # 高成本级别空跑样本惩罚(0.1倍)
    reward = tf.where(
        is_high_cost_level_and_empty_run,
        reward * 0.1,
        reward
    )
    # 中等成本级别空跑样本惩罚(0.1倍)
    reward = tf.where(
        is_medium_cost_level_and_empty_run,
        reward * 0.1,
        reward
    )
    # 高成本级别高估样本轻微惩罚(0.8倍)
    reward = tf.where(
        is_high_cost_level_and_high_bill,
        reward * 0.8,
        reward
    )
    # 中等成本级别高估样本轻微惩罚(0.8倍)
    reward = tf.where(
        is_medium_cost_level_and_high_bill,
        reward * 0.8,
        reward
    )
    
    return reward

def get_pow_w(task_name, is_auto_type, logit_pre, use_trans_learning=False):
    """
    获取不同任务类型的正样本权重参数
    功能：
        根据任务类型、是否为自动类型以及是否使用迁移学习, 返回相应的正样本权重
        用于处理类别不平衡问题
    输入：
        task_name: 字符串, 任务名称, 支持"convert"、"active"和"cost"
        is_auto_type: 张量, 指示样本是否为自动类型
        logit_pre: 张量, 模型预测的logits值, 用于确定权重的形状
        use_trans_learning: 布尔值, 是否使用迁移学习, 默认为False
    输出：
        张量, 与logit_pre形状相同的权重值
    异常：
        ValueError: 当提供不支持的任务名称时抛出
    """
    # 初始化默认权重
    pos_w = tf.constant(1.0)
    
    # 根据任务类型和参数设置不同的正样本权重
    if task_name == "convert":  # 转化任务
        if use_trans_learning:
            # 迁移学习模式下使用固定权重
            pos_w = tf.constant(12.4)
        else:
            # 根据是否为自动类型设置不同权重
            pos_w = tf.where(
                tf.equal(is_auto_type, 0),  # 非自动类型
                tf.ones_like(logit_pre, tf.float32) * 6.2,
                tf.ones_like(logit_pre, tf.float32) * 17  # 自动类型
            )
    elif task_name == "active":  # 活跃度任务
        if use_trans_learning:
            pos_w = tf.constant(3.2)
        else:
            pos_w = tf.where(
                tf.equal(is_auto_type, 0),
                tf.ones_like(logit_pre, tf.float32) * 2.4,
                tf.ones_like(logit_pre, tf.float32) * 3.2
            )
    elif task_name == "cost":  # 成本任务
        if use_trans_learning:
            pos_w = tf.constant(4.3)
        else:
            pos_w = tf.where(
                tf.equal(is_auto_type, 0),
                tf.ones_like(logit_pre, tf.float32) * 1.8,
                tf.ones_like(logit_pre, tf.float32) * 3.4
            )
    else:
        # 抛出不支持的任务名称异常
        raise ValueError("unk task: {}".format(task_name))
    
    return pos_w
