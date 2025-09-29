# encoding=utf-8
import os
import logging
import tensorflow as tf
import numpy as np
import json

from lagrange_lite.sparse import data, parser, readers
from lagrange_lite.common import metrics
import pickle
if 'TF_CONFIG' not in os.environ:
    _TASK_NAME = 'chief-0'
else:
    tf_config = json.loads(os.environ['TF_CONFIG'])
    cur_task = tf_config['task']
    _TASK_NAME = '%s-%d' % (cur_task['type'], cur_task['index'])

METRICS_TAG_KV = {'task': _TASK_NAME}

def create_instance_dataset(
        raw_paths, sparse_keys=[], num_shards=1, shard_id=0, batch_size=1024,
        dense_features={}, shuffle_buffer_size=64 * 1024 * 1024, cycle_length=4,
        block_length=2, num_parallel_maps=None, n_epochs=None,
        num_prefetch=-1,
        is_auto_type=-1,
        only_netservice=False
):
    """
    功能: 创建并配置TensorFlow数据集实例, 用于模型训练或评估
    输入: 
        raw_paths: 字符串或字符串列表, 表示数据源文件路径
        sparse_keys: 列表, 稀疏特征的键名列表
        num_shards: 整数, 表示将数据集分成的分片数量
        shard_id: 整数, 表示当前使用的分片ID
        batch_size: 整数, 批处理大小
        dense_features: 字典, 定义稠密特征的结构
        shuffle_buffer_size: 整数, 随机打乱的缓冲区大小
        cycle_length: 整数, 并行读取文件的数量
        block_length: 整数, 从每个文件读取的连续记录数
        num_parallel_maps: 整数或None, 并行处理的数量
        n_epochs: 整数或None, 数据集重复的次数
        num_prefetch: 整数, 预取的批次数
        is_auto_type: 整数, 过滤特定广告类型的条件
        only_netservice: 布尔值, 是否仅使用非特定广告ID的数据
    输出: 
        TensorFlow数据集对象, 已经配置好数据处理流程
    """
    # 展开并排序原始文件路径
    expanded_and_sorted = data.expand_and_sort_paths(raw_paths)
    # 随机打乱文件顺序
    np.random.shuffle(expanded_and_sorted)

    # 创建文件路径数据集
    files_to_read = tf.data.Dataset.from_tensor_slices(expanded_and_sorted)
    # 如果需要分片, 执行分片操作
    if num_shards > 1:
        logging.info("Dataset shard is: %d/%d", shard_id, num_shards)
        files_to_read = files_to_read.shard(num_shards, shard_id)

    def parse_fn(serialized):
        """解析单条序列化记录的函数"""
        # 解析序列化数据, 提取特征和标签
        features = parser.parse_single_instance(
            serialized,
            sparse_keys=sparse_keys,  # 稀疏特征键
            fields={
                'label': tf.io.FixedLenFeature(shape=[7], dtype=tf.float32, default_value=[0.0]*7)  # 标签定义
            },
            lineid_fields={  # 行ID相关字段
                'req_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
                'customer_id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=[0]),
                'advertiser_id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=[0]),
                'external_action': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
                'deep_external_action': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=[0]),
                'app_package': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
                'target_app_package': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
                'ltr_rank_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
                'deep_bid_type': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=[0]),
                'ad_id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=[0])
            },
            dense_fields=dense_features  # 稠密特征定义
        )
        
        # 从特征字典中取出标签并返回
        label = features.pop('label')
        return features, label

    def filter_fn_v2(features, label):
        """过滤特定广告类型的函数"""
        return tf.reshape(tf.equal(features['fc_dense_auto_ad_type'], is_auto_type), [])
    
    def filter_fn_v3(features, label):
        """过滤特定广告ID的函数"""
        return tf.reshape(tf.not_equal(features['ad_id'], 1913), [])

    # 创建并配置数据集处理流程
    dataset = files_to_read.interleave(
        # 并行读取多个文件
        map_func=lambda one_path: readers.InstanceDataset(
            one_path,
            use_snappy=True,  # 使用snappy压缩
            has_prefix=True,  # 文件有前缀
            has_sort_id=True,  # 有排序ID
            has_kafka_dump=False  # 非Kafka dump格式
        ),
        cycle_length=cycle_length,  # 并行读取文件数
        block_length=block_length,  # 每个文件读取的连续记录数
        num_parallel_calls=cycle_length,
    ).map(
        # 解析数据
        parse_fn,
        num_parallel_calls=(tf.data.experimental.AUTOTUNE if num_parallel_maps is None else num_parallel_maps)
    ).shuffle(buffer_size=shuffle_buffer_size)  # 打乱数据
    print(" lzx_dataset dataset example:", dataset.take(1))
    # 根据条件过滤数据集
    if is_auto_type >= 0:
        dataset = dataset.filter(filter_fn_v2)
    
    if only_netservice:
        dataset = dataset.filter(filter_fn_v3)

    # 批处理数据
    dataset = dataset.batch(batch_size)

    # 设置数据集重复次数
    if n_epochs is not None and n_epochs > 0:
        dataset = dataset.repeat(n_epochs)

    # 设置预取数量, 优化性能
    if num_prefetch > 0:
        dataset = dataset.prefetch(num_prefetch)
    elif num_prefetch == 0:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


class CustomMetricHook(tf.estimator.SessionRunHook):
    """
    功能: TensorFlow评估器的自定义指标钩子, 用于记录和计算批次级别的指标
    输入: 
        metric_tensors: 字典, 键为指标名称, 值为对应的标量张量
        log_steps: 整数, 日志记录的步长间隔
        ema_decay: 浮点数, 指数移动平均的衰减系数
    输出: 
        无直接输出, 作为钩子集成到TensorFlow训练过程中
    """

    def __init__(self, metric_tensors, log_steps=1, ema_decay=0.999):
        """初始化指标钩子"""
        # 验证输入的指标张量是否合法
        for name in metric_tensors:
            tensor = metric_tensors[name]
            if len(tensor.shape.dims) > 0:
                raise ValueError('The metric tensor should be a scalar!')
            if tensor.dtype.base_dtype not in (tf.float32, tf.int32):
                raise ValueError('The dtype of a metric tensor should be either tf.float or tf.int32!')
        if len(metric_tensors) == 0:
            raise ValueError('At least one metric tensor should be offered!')
        assert (log_steps > 0)
        
        # 初始化成员变量
        self._metric_tensors = metric_tensors
        self._log_steps = log_steps
        self._step = 0
        self._ema_decay = ema_decay
        self._values = {name: [] for name in metric_tensors}  # 存储每个指标的历史值
        self._ema = {name: 0.0 for name in metric_tensors}  # 存储每个指标的指数移动平均值

    def before_run(self, run_context):
        """在会话运行前调用, 指定要获取的张量"""
        return tf.estimator.SessionRunArgs(self._metric_tensors)

    def after_run(self, run_context, run_value):
        """在会话运行后调用, 处理指标结果并记录日志"""
        metric_values = run_value.results
        
        # 处理每个指标的值
        for name in metric_values:
            # 存储指标值
            metrics.emit_store(name, float(metric_values[name]), tagkv=METRICS_TAG_KV)
            self._values[name].append(float(metric_values[name]))
            # 更新指数移动平均值
            self._ema[name] = self._ema_decay * self._ema[name] + (1.0 - self._ema_decay) * float(metric_values[name])
        
        self._step += 1
        # 按指定步长记录日志
        if self._step % self._log_steps == 0:
            logging.info('Step[{:d}]:'.format(self._step) + ''.join([
                '\t' + name + ': (batch: ' + str(metric_values[name]) +
                ', mean: ' + str(np.mean(self._values[name])) +
                ', ema: ' + str(self._ema[name]) + ')' for name in metric_values]))


if __name__ == "__main__":
    feat_names = {'fc_aid_shadow_cost', 'fc_aid_shadow_cost_last',
                  'fc_dense_external_action_last', 'fc_dense_ad_cid_shadow_num_max_strategy',
                  'fc_dense_ad_cid_shadow_num_max_strategy_last'}
    dense_features = {fc: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=[0]) for fc in
                      sorted(list(feat_names))}