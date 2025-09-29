# encoding:utf-8

import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
import math

def data_compress(feat_val, need_compress=True):
    """
    数据压缩函数
    功能：
        将输入的特征值转换为浮点数, 并根据需要进行log1p压缩, 确保输出值非负
    输入：
        feat_val: 输入特征值, 可以是张量或数值
        need_compress: 布尔值, 是否需要进行log1p压缩, 默认为True
    输出：
        压缩后的特征张量, 数据类型为tf.float32
    """
    if need_compress:
        # 将特征值转换为float32类型, 取与0的最大值确保非负, 然后应用log1p压缩
        return tf.math.log1p(tf.maximum(tf.cast(feat_val, tf.float32), 0.0))
    return tf.maximum(tf.cast(feat_val, tf.float32), 0.0)

def bucket_single_feat(feat_val, bucket_points, fc="fc", dim=1, prefix="", suffix="", need_summary=False):
    """
    单特征分桶与嵌入函数
    功能：
        将连续特征值按照指定的分桶点进行离散化, 然后通过嵌入层将离散值转换为低维向量表示
    输入：
        feat_val: 输入的连续特征值张量
        bucket_points: 分桶边界点列表
        fc: 特征名称, 用于变量命名, 默认为"fc"
        dim: 嵌入向量维度, 默认为1
        prefix: 变量名前缀, 默认为空字符串
        suffix: 变量名后缀, 默认为空字符串
        need_summary: 是否记录摘要信息, 默认为False
    输出：
        嵌入向量张量, 形状为[batch_size, dim]
    """
    bucket_num = len(bucket_points) + 1
    # 对输入特征进行分桶处理, 得到每个样本对应的桶索引
    bucket_info = gen_math_ops.bucketize(
        input=feat_val, boundaries=bucket_points
    )
    
    # 处理特征名称, 添加前缀和后缀
    if prefix:
        fc = prefix + fc
    if suffix:
        fc = fc + suffix
    
    # 如果需要记录摘要信息
    if need_summary:
        tf.summary.histogram(fc + '_histogram', bucket_info)  # 记录桶索引的直方图
        tf.summary.scalar(fc + 'bucket_mean', tf.reduce_mean(tf.cast(bucket_info, tf.float32)))  # 记录桶索引的均值
    
    # 创建嵌入权重矩阵, 形状为[bucket_num, dim]
    weights = tf.get_variable(name=fc + '_weights',
                          shape=[bucket_num, dim],
                          initializer=tf.random_uniform_initializer(-0.0018125, 0.0018125))
    
    # 根据桶索引从权重矩阵中获取对应的嵌入向量
    emb_tensor = tf.gather(weights, bucket_info, axis=0)
    # 调整嵌入向量形状为[batch_size, dim]
    emb_tensor = tf.reshape(emb_tensor, shape=[-1, dim])
    
    return emb_tensor

def _gen_center(buckets=None, bucket_lower_bound=0.0, bucket_upper_bound=70.1):
    """
    生成分桶中心点的辅助函数
    功能：
        给定分桶边界点, 计算每个分桶的中心点坐标
    输入：
        buckets: 分桶边界点列表, 如果为None则使用默认值
        bucket_lower_bound: 分桶的下界值, 默认为0.0
        bucket_upper_bound: 分桶的上界值, 默认为70.1
    输出：
        分桶中心点列表
    """
    # 如果没有提供分桶边界点, 则使用默认的分桶点
    if not buckets:
        buckets = [0.1, 2.1, 4.1, 6.1, 8.1, 10.1, 12.1, 14.1, 16.1, 18.1, 20.1, 22.1, 24.1, 26.1, 28.1, 30.1, 32.1, 34.1, 36.1, 38.1, 40.1, 42.1, 44.1, 46.1, 48.1, 50.1, 52.1, 54.1, 56.1, 58.1, 60.1]
    
    # 添加上下界到分桶边界点列表
    buckets = [bucket_lower_bound] + buckets + [bucket_upper_bound]
    mid_buckets = list()
    
    # 计算每个相邻分桶边界点的中点, 作为分桶中心点
    for i in range(0, len(buckets) - 1):
        mid_buckets.append((buckets[i] + buckets[i + 1]) / 2.0)
    
    return mid_buckets

# ad_cnt key-value mem network
def bucket_single_feat_semantic(feat_val, bucket_points, fc="fc",
                                dim=1, prefix="", suffix="", need_summary=False, method='auto',temp=0.8, bucket_lower_bound=0.0, bucket_upper_bound=70.1):
    """
    语义感知的单特征分桶与嵌入函数
    功能：
        基于注意力机制将连续特征值映射到分桶嵌入向量上, 支持多种注意力计算方式
    输入：
        feat_val: 输入的连续特征值张量
        bucket_points: 分桶边界点列表
        fc: 特征名称, 用于变量命名, 默认为"fc"
        dim: 嵌入向量维度, 默认为1
        prefix: 变量名前缀, 默认为空字符串
        suffix: 变量名后缀, 默认为空字符串
        need_summary: 是否记录摘要信息, 默认为False
        method: 注意力计算方法, 可选'auto'或'dis', 默认为'auto'
        temp: 温度参数, 用于'dis'方法中的softmax归一化, 默认为0.8
        bucket_lower_bound: 分桶下界, 默认为0.0
        bucket_upper_bound: 分桶上界, 默认为70.1
    输出：
        语义感知的嵌入向量张量, 形状为[batch_size, dim]
    """
    bucket_num = len(bucket_points) + 1
    # 对输入特征进行分桶处理
    bucket_info = gen_math_ops.bucketize(
        input=feat_val, boundaries=bucket_points
    )
    
    if prefix:
        fc = prefix + fc
    if suffix:
        fc = fc + suffix
    
    if need_summary:
        tf.summary.histogram(fc + '_histogram', bucket_info)  # 记录桶索引的直方图
        tf.summary.scalar(fc + 'bucket_mean', tf.reduce_mean(tf.cast(bucket_info, tf.float32)))  # 记录桶索引的均值
    
    # 创建嵌入权重矩阵
    weights = tf.get_variable(name=fc + '_weights',
                              shape=[bucket_num, dim],
                              initializer=tf.random_uniform_initializer(-0.0018125, 0.0018125))
    
    # 特征值转换为浮点型并进行log1p压缩
    feat_val = tf.cast(feat_val, tf.float32)
    log_feat_val = tf.math.log1p(feat_val)
    
    # 创建线性变换权重
    h_1 = tf.get_variable(name=fc + '_h1',
                          shape=[1, bucket_num],
                          initializer=tf.random_uniform_initializer(-0.0018125, 0.0018125))
    
    # 根据指定的方法计算注意力权重
    if method == 'auto':
        # 自动学习的注意力模式：通过神经网络学习特征与分桶之间的关系
        # 将log压缩的特征映射到分桶空间
        log_feat_val_map = tf.nn.leaky_relu(tf.einsum(
            "ij,jk->ik", log_feat_val, h_1)
        )
        
        # 创建变换矩阵
        w_1 = tf.get_variable(name=fc + '_W1',
                              shape=[bucket_num, bucket_num],
                              initializer=tf.random_uniform_initializer(-0.0018125, 0.0018125))
        
        # 计算注意力得分的前体, 添加残差连接
        attn_pre = tf.einsum(
            'ij,jk->ik', log_feat_val_map, w_1
        ) + 0.1 * log_feat_val_map
        
        # 应用温度参数缩放
        t = 0.2
        attn_pre = attn_pre / t
        tf.summary.histogram(fc + '{}_attn_post'.format(method), attn_pre)  # 记录注意力得分直方图
        
        # 通过softmax归一化得到注意力权重
        attn_w = tf.nn.softmax(
            attn_pre, axis=-1
        )
        
    elif method == 'dis':
        # 基于距离的注意力模式：计算特征值与分桶中心点的距离
        # 生成分桶中心点
        mid_buckets = _gen_center(bucket_points, bucket_lower_bound=bucket_lower_bound, bucket_upper_bound=bucket_upper_bound)
        assert len(mid_buckets) == bucket_num, "分桶中心点数量与分桶数量不匹配: {} != {}".format(len(mid_buckets), bucket_num)
        
        # 将中心点转换为常量张量
        mid_constants = tf.reshape(tf.constant(mid_buckets, dtype=tf.float32), [1, len(mid_buckets)])
        feat_val = tf.reshape(feat_val, [-1, 1])
        
        # 将特征值张量扩展为与中心点数量匹配的形状
        feat_val = tf.tile(feat_val, [1, len(mid_buckets)])
        
        # 计算特征值与每个分桶中心点的绝对距离
        dis_diff = tf.abs(feat_val - mid_constants)
        
        # 根据距离计算注意力得分, 距离越小得分越高
        t = temp
        attn_pre = -1 * dis_diff / t
        tf.summary.histogram(fc + '{}_attn_post'.format(method), attn_pre)  # 记录注意力得分直方图
        
        # 通过softmax归一化得到注意力权重
        attn_w = tf.nn.softmax(attn_pre, axis=-1)
        
    else:
        raise ValueError('未知的注意力计算方法: {}'.format(method))
    
    # 通过注意力权重对嵌入向量进行加权求和, 得到最终的语义感知嵌入向量
    emb_tensor = tf.einsum('ij,jk->ik', attn_w, weights)
    
    return emb_tensor

def _debug_feat(feat, msg):
    """
    调试特征的辅助函数
    功能：
        在TensorFlow计算图执行时打印特征的值和形状, 用于调试
    
    输入：
        feat: 需要调试的张量
        msg: 打印消息前缀
    输出：
        处理后的特征张量(与输入相同)
    """
    # 使用tf.Print在图执行时打印特征值和形状信息\# summarize=-1表示打印所有元素
    feat = tf.Print(feat, [feat, tf.shape(feat)], message=msg, summarize=-1)
    return feat

def _pad_feat(features, fc, pad_feat):
    """
    特征填充辅助函数
    功能：
        从特征字典中获取指定特征, 如果特征不存在则返回与参考特征形状相同的零张量
    输入：
        features: 特征字典, 键为特征名称, 值为特征张量
        fc: 需要获取的特征名称
        pad_feat: 用于确定返回零张量形状的参考特征张量
    输出：
        获取的特征张量或形状匹配的零张量
    """
    # 检查特征是否存在于字典中
    if fc not in features:
        # 如果不存在, 返回与参考特征形状相同的零张量
        return tf.zeros_like(pad_feat)
    # 如果存在, 返回该特征张量
    return features[fc]

def bucket_idx(feat_val, bucket_points):
    """
    特征分桶索引计算函数
    功能：
        将连续特征值按照指定的分桶点进行离散化, 返回每个样本对应的桶索引
    输入：
        feat_val: 输入的连续特征值张量
        bucket_points: 分桶边界点列表, 按升序排列
    输出：
        分桶索引张量, 形状与输入特征相同, 值为整数桶编号
    """
    # 使用TensorFlow的bucketize操作计算特征值对应的桶索引
    # 桶编号从0开始, 按照输入边界点划分为多个区间
    bucket_info = gen_math_ops.bucketize(
        input=feat_val, boundaries=bucket_points
    )
    return bucket_info

def bucket_time_delta_feats(features, cur_time_delta_feat, cur_time_delta_name, feat_to_bucket, dim=1, filter_shadow=False):
    """
    时间差特征处理与嵌入函数
    功能：
        对时间差特征进行处理并生成对应的嵌入向量, 支持过滤shadow特征
    输入：
        features: 特征字典, 包含所有输入特征
        cur_time_delta_feat: 时间差特征名称列表, 按每3个特征为一组
        cur_time_delta_name: 时间差特征组名称列表
        feat_to_bucket: 特征分桶点字典, 键为特征名称, 值为分桶边界点
        dim: 嵌入向量维度, 默认为1
        filter_shadow: 是否过滤包含"shadow"的特征, 默认为False
    输出：
        state_embeddings: 嵌入向量列表
        state_embeddings_names: 对应嵌入向量的特征名称列表
        filtered_shadow_feats: 被过滤掉的shadow特征列表
    """
    state_embeddings = list()  # 存储生成的嵌入向量
    state_embeddings_names = list()  # 存储嵌入向量对应的特征名称
    
    # 验证时间差特征数量是3的倍数
    assert len(cur_time_delta_feat) % 3 == 0, "时间差特征数量必须是3的倍数: {} error".format(len(cur_time_delta_feat))
    # 验证特征组名称数量与特征组数匹配
    assert len(cur_time_delta_feat) // 3 == len(
        cur_time_delta_name), "特征数量与特征组名称数量不匹配: cur_time_delta_feat: {} != cur_time_delta_name: 3 *  {}".format(
        len(cur_time_delta_feat), len(cur_time_delta_name))
    
    feat_dict = dict()
    # 每3个特征作为一组进行处理
    for idx in range(0, len(cur_time_delta_feat), 3):
        feat_name = [cur_time_delta_feat[idx], cur_time_delta_feat[idx+1], cur_time_delta_feat[idx+2]]
        # 计算时间差特征并更新特征字典, 使用[1.0, 3.0, 7.0]作为时间窗口
        feat_dict.update(time_delta_feats(features, feat_name, [1.0, 3.0, 7.0], name=cur_time_delta_name[idx // 3]))

    filtered_shadow_feats = list()
    # 遍历所有特征, 生成嵌入向量
    for feat_name, feat_tensor in feat_dict.items():
        # 如果需要过滤shadow特征且当前特征包含"shadow"
        if filter_shadow and "shadow" in feat_name:
            filtered_shadow_feats.append(feat_name)
            continue
        
        # 将特征转换为float32类型
        feat_val = tf.cast(feat_tensor, tf.float32)
        # 记录特征均值的平方根到摘要
        tf.summary.scalar('raw_feat_mean_' + feat_name,
                          tf.sqrt(tf.reduce_mean(feat_val)))
        # 为特征生成嵌入向量并添加到列表
        state_embeddings.append(
            bucket_single_feat(feat_val, feat_to_bucket[feat_name], fc=feat_name, dim=dim)
        )
        state_embeddings_names.append(feat_name)

    return state_embeddings, state_embeddings_names, filtered_shadow_feats

def bucket_float_feats(features, cur_float_feat, feat_to_bucket, dim=1, filter_shadow=False):
    """
    浮点数特征处理与嵌入函数
    功能：
        对浮点数特征(如pctr、pcvr等)进行处理并生成对应的嵌入向量, 支持过滤shadow特征
    输入：
        features: 特征字典, 包含所有输入特征
        cur_float_feat: 浮点数特征名称列表
        feat_to_bucket: 特征分桶点字典, 键为特征名称, 值为分桶边界点
        dim: 嵌入向量维度, 默认为1
        filter_shadow: 是否过滤包含"shadow"的特征, 默认为False
    输出：
        state_embeddings: 嵌入向量列表
        state_embeddings_names: 对应嵌入向量的特征名称列表
        filtered_shadow_feats: 被过滤掉的shadow特征列表
    """
    state_embeddings = list()  # 存储生成的嵌入向量
    state_embeddings_names = list()  # 存储嵌入向量对应的特征名称
    filtered_shadow_feats = list()  # 存储被过滤掉的shadow特征
    
    # 遍历所有浮点数特征
    for feat_name in cur_float_feat:
        # 如果需要过滤shadow特征且当前特征包含"shadow"
        if filter_shadow and "shadow" in feat_name:
            filtered_shadow_feats.append(feat_name)
            continue
        
        # 获取特征值
        feat_val = features[feat_name]
        # 记录特征均值的平方根到摘要
        tf.summary.scalar('raw_feat_mean_' + feat_name,
                          tf.sqrt(tf.reduce_mean(feat_val)))
        # 为特征生成嵌入向量并添加到列表
        state_embeddings.append(
            bucket_single_feat(feat_val, feat_to_bucket[feat_name], fc=feat_name, dim=dim)
        )
        state_embeddings_names.append(feat_name)

    return state_embeddings, state_embeddings_names, filtered_shadow_feats

def _softmax_with_mask_(logits, masks, epsilon=1e-6):
    """
    带掩码的softmax操作函数
    功能：
        对输入的logits进行softmax归一化, 同时考虑掩码值, 确保掩码位置不参与归一化
    输入：
        logits: 输入的logits张量, 形状为[B, h, F, T]
               B: 批次大小, h: 头数, F: 查询序列长度, T: 键值序列长度
        masks: 掩码张量, 形状为[B, h, F, T], 值为True/False表示有效/无效位置
        epsilon: 防止除零错误的小值, 默认为1e-6
    输出：
        归一化后的概率分布张量, 形状与输入相同[B, h, F, T]
    """
    # 计算最后一个维度上的最大值, 用于数值稳定性
    max_logits = tf.reduce_max(logits, axis=-1, keepdims=True)
    # 减去最大值后计算指数, 增强数值稳定性
    safe_exp_logits = tf.exp(logits - max_logits)
    
    # 如果提供了掩码, 则将无效位置的指数值置为0
    if masks is not None:
        safe_exp_logits = tf.where(masks, safe_exp_logits, tf.zeros_like(safe_exp_logits))
    
    # 计算softmax归一化, 添加epsilon防止除零错误
    return safe_exp_logits / (tf.reduce_sum(safe_exp_logits, axis=-1, keepdims=True) + epsilon)

def _attention_(queries, keys, values, masks=None, name_suffix=""):
    """
    注意力机制实现函数
    功能：
        计算查询向量(queries)对键向量(keys)的注意力权重, 并使用权重对值向量(values)进行加权求和
    输入：
        queries: 查询张量, 形状为[B, h, F, D]
                B: 批次大小, h: 头数, F: 查询序列长度, D: 特征维度
        keys: 键张量, 形状为[B, h, T, D]
        values: 值张量, 形状为[B, h, T, D]
        masks: 掩码张量, 形状为[B, h, T], 默认为None
        name_suffix: 摘要名称的后缀, 用于区分不同的注意力计算
    输出：
        注意力加权后的输出张量, 形状为[B, h, F, D]
    """
    # 获取特征维度D, 用于缩放
    dim = queries.get_shape().as_list()[-1]
    # 计算注意力得分：查询与键的点积除以维度的平方根(缩放点积注意力)
    attn_logits = tf.matmul(queries, keys, transpose_b=True) / math.sqrt(dim)  # [B, h, F, T]

    # 处理掩码：如果提供了掩码, 则调整其形状以匹配注意力得分
    if masks is not None:
        from_seq_len = tf.shape(attn_logits)[-2]  # 获取查询序列长度F
        masks = tf.expand_dims(masks, axis=[-2])  # [B, h, 1, T] 将掩码扩展一个维度
        masks = tf.tile(masks, [1, 1, from_seq_len, 1])  # [B, h, F, T] 复制掩码以匹配注意力得分的形状

    # 计算注意力权重：对得分进行带掩码的softmax归一化
    attn_weights = _softmax_with_mask_(attn_logits, masks)  # [B, h, F, T]

    # 提取有效位置的注意力权重并记录直方图
    valid_attn_w = tf.boolean_mask(
        attn_weights, masks
    )
    tf.summary.histogram(
        "valid_attn_w_{}".format(name_suffix), valid_attn_w
    )

    # 计算注意力输出：使用注意力权重对值向量进行加权求和
    attn_output = tf.matmul(attn_weights, values)  # [B, h, F, D]

    return attn_output

def multi_head_attention(queries, keys, values, qk_dim, v_dim, head_num, masks=None, pad_len=20, name_suffix=""):
    """
    多头注意力机制实现函数
    功能：
        将输入向量通过多头注意力机制进行处理, 实现不同子空间的特征提取和信息融合
    输入：
        queries: 查询张量, 形状为[B, F, H]
                B: 批次大小, F: 查询序列长度, H: 输入特征维度
        keys: 键张量, 形状为[B, T, H]
        values: 值张量, 形状为[B, T, H]
        qk_dim: 查询和键的变换维度
        v_dim: 值的变换维度
        head_num: 注意力头的数量
        masks: 掩码张量, 形状为[B, T], 默认为None
        pad_len: 序列填充长度, 默认为20
        name_suffix: 摘要名称的后缀, 默认为空字符串
    输出：
        多头注意力处理后的输出张量, 形状为[B, F, H]
    """

    def forward(x, layer_name, dim):
        """前向线性变换函数"""
        dense_layer = tf.layers.dense(
            inputs=x,
            units=dim,
            use_bias=False,
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
            activation=None,
            name=layer_name)
        return dense_layer

    def split_head(x, dim, h_num, name=""):
        """将输入张量分割成多个头"""
        # 验证维度可以被头数整除
        assert dim % h_num == 0, "维度必须能被头数整除: dim: {}\thead_num: {}".format(dim, h_num)
        dim_per_head = int(dim / h_num)  # 每个头的维度
        # 重塑并转置张量, 将头维度放在第二维
        return tf.transpose(
            tf.reshape(x, [-1, pad_len, h_num, dim_per_head], name=name),
            perm=[0, 2, 1, 3])  # 输出形状: [B, h, F, D_per_head]

    def merge_head(x, h_num, dim_per_head, name=""):
        """将多个头的输出合并成一个张量"""
        # 转置并重塑张量, 将多头结果合并
        return tf.reshape(
            tf.transpose(x, perm=[0, 2, 1, 3]),
            [-1, pad_len, h_num * dim_per_head], name=name)  # 输出形状: [B, F, h*D_per_head]

    # 对查询、键和值进行线性变换
    queries = forward(queries, "query_linear_" + name_suffix, qk_dim)  # [B, F, qk_dim]
    keys = forward(keys, "key_linear_" + name_suffix, qk_dim)  # [B, T, qk_dim]
    values = forward(values, "value_linear_" + name_suffix, v_dim)  # [B, T, v_dim]

    # 将查询、键和值分割成多个头
    queries = split_head(queries, qk_dim, head_num, "queries_" + name_suffix)  # [B, h, F, D1]
    keys = split_head(keys, qk_dim, head_num, "keys_" + name_suffix)  # [B, h, T, D1]
    values = split_head(values, v_dim, head_num, "values_" + name_suffix)  # [B, h, T, D2]

    # 处理掩码：扩展掩码维度以匹配多头结构
    if masks is not None:
        masks = tf.expand_dims(masks, axis=[1])  # [B, 1, T] 将掩码扩展一个维度
        masks = tf.tile(masks, [1, head_num, 1])  # [B, h, T] 为每个头复制掩码

    # 计算多头注意力输出
    attn_output = _attention_(queries, keys, values, masks, name_suffix=name_suffix)  # [B, h, F, D2]

    # 将多头结果合并成一个张量
    attn_output = merge_head(attn_output, head_num, int(v_dim / head_num), "attn_merge_" + name_suffix)  # [B, F, h*D2]

    return attn_output

def din_seq_model(query_emb, valid_masks, seq_embs, emb_size, name):
    """
    基于DIN (Deep Interest Network) 的序列建模函数
    功能：
        对素材序列进行差异化建模, 通过掩码和平均值池化提取有效序列特征
    输入：
        query_emb: 查询嵌入向量, 形状为[batch, emb_size]
        valid_masks: 有效掩码列表, 每个元素形状为[batch, pad_len]
                     列表长度等于特征数量
        seq_embs: 序列嵌入列表, 每个元素形状为[batch, pad_len, emb_size]
                  列表长度等于特征数量
        emb_size: 嵌入向量维度
        name: 模型名称, 用于摘要命名
    输出：
        masked_vmid_seq: 掩码池化后的序列特征, 形状为[batch, emb * num_of_feat]
        valid_num: 每个样本的有效序列长度, 形状为[batch, 1]
    """
    # 验证掩码列表和序列嵌入列表长度匹配
    assert len(valid_masks) == len(seq_embs), "掩码数量与序列嵌入数量不匹配: len_of_masks: {} != seq_embs: {}".format(
        len(valid_masks), len(seq_embs)
    )
    
    # 将每个掩码扩展一个维度并转换为浮点型
    # [b, pad_len, 1] * len_of_feat
    valid_masks = [tf.cast(tf.expand_dims(m, axis=-1), tf.float32) for m in valid_masks]
    
    # 沿特征维度拼接所有掩码
    # [b, pad_len, len_of_feat]
    valid_mask_concat = tf.concat(valid_masks, axis=-1)
    
    # 计算每个位置是否至少有一个特征有效
    # [b, pad_len]
    valid_num = tf.reduce_max(
        valid_mask_concat, axis=-1
    )
    
    # 计算每个样本的有效序列长度
    # [b, 1]
    valid_num = tf.reduce_sum(
        valid_num, axis=1, keepdims=True
    )
    
    # 将每个掩码扩展到嵌入维度, 然后沿特征维度拼接
    # [b, pad_len, emb * len_of_feat]
    valid_mask_tensor = tf.concat(
        [tf.tile(m, [1, 1, emb_size]) for m in valid_masks], axis=-1
    )
    
    # 记录有效序列长度的直方图
    tf.summary.histogram(
        "valid_num_{}".format(name),
        valid_num
    )
    
    # 沿嵌入维度拼接所有序列嵌入
    # [b, pad_len, emb * num_of_feat]
    vmid_seq_embedding = tf.concat(
        seq_embs, axis=2
    )
    
    # 计算每个样本在每个特征上的有效位置数量
    # [b, emb * num_of_feat]
    valid_num_seq = tf.reduce_sum(
        valid_mask_tensor, axis=1
    )
    
    # 对序列嵌入应用掩码并沿序列长度维度求和
    masked_vmid_seq = tf.reduce_sum(vmid_seq_embedding * valid_mask_tensor, axis=1)
    
    # 计算掩码后的平均值, 使用divide_no_nan避免除零错误
    # [b, emb * num_of_feat]
    masked_vmid_seq = tf.math.divide_no_nan(masked_vmid_seq, valid_num_seq)
    
    return masked_vmid_seq, valid_num

def vmid_seq_model(multi_vmid_embeddings, valid_masks, qk_emb_size, emb_size, name, head_num=4, pad_len=20):
    """
    基于多头注意力机制的素材序列建模函数
    功能：
        对多个素材序列进行建模, 通过多头自注意力机制提取序列特征
    输入：
        multi_vmid_embeddings: 多个素材嵌入列表, 每个元素形状为[batch, pad_len, emb_size]
        valid_masks: 有效掩码列表, 每个元素形状为[batch, pad_len]
        qk_emb_size: 查询和键的嵌入维度
        emb_size: 输出嵌入维度, 通常为3 * 特征数量
        name: 模型名称, 用于摘要命名
        head_num: 注意力头的数量, 默认为4
        pad_len: 序列填充长度, 默认为20
    输出：
        atten_vmid_emb: 注意力加权后的序列特征, 形状为[batch, emb_size]
        valid_num: 每个样本的有效序列长度, 形状为[batch, 1]
    """
    # 将掩码列表堆叠成张量
    # [b, num_of_feat, pad_len]
    valid_mask_tensor = tf.stack(
        valid_masks, axis=1
    )
    
    # 计算每个位置是否至少有一个特征有效
    # [b, pad_len]
    valid_mask_tensor = tf.reduce_max(
        tf.cast(valid_mask_tensor, tf.int32), axis=1
    )
    
    # 计算每个样本的有效序列长度
    valid_num = tf.reduce_sum(
        tf.cast(valid_mask_tensor, tf.int32), axis=1, keepdims=True
    )
    
    # 记录有效序列长度的直方图
    tf.summary.histogram(
        "valid_num_{}".format(name),
        valid_num
    )
    
    # 沿嵌入维度拼接所有素材嵌入
    # [b, pad_len, emb * num_of_feat]
    vmid_seq_embedding = tf.concat(
        multi_vmid_embeddings, axis=2
    )
    
    # 将掩码转换为布尔型
    valid_mask_tensor = tf.cast(valid_mask_tensor, tf.bool)
    
    # 应用多头自注意力机制
    # 查询、键、值都使用同一个序列嵌入
    atten_vmid_emb = multi_head_attention(
        vmid_seq_embedding, vmid_seq_embedding, vmid_seq_embedding,
        int(qk_emb_size), int(emb_size), int(head_num), masks=valid_mask_tensor, name_suffix=name, pad_len=pad_len)
    
    # 再次应用掩码, 将无效位置的注意力输出置为0
    atten_vmid_emb = tf.where(
        tf.tile(tf.expand_dims(valid_mask_tensor, axis=2), [1, 1, int(emb_size)]),
        atten_vmid_emb,
        tf.zeros_like(atten_vmid_emb)
    )
    
    # 沿序列长度维度求和
    atten_vmid_emb = tf.reduce_sum(
        atten_vmid_emb, axis=1
    )
    
    # 计算掩码后的平均值, 对于有效序列长度为0的样本, 输出0
    atten_vmid_emb = tf.where(
        tf.tile(valid_num > 0, [1, int(emb_size)]),
        atten_vmid_emb / tf.cast(valid_num, tf.float32),
        tf.zeros_like(atten_vmid_emb, tf.float32)
    )
    
    return atten_vmid_emb, valid_num

def gen_delta_feat(features, feat_type="", delta_type_feat=None, feat_bucket=None):
    """
    生成delta特征的函数(函数主体)
    功能：
        生成两个时间窗口特征之间的差值特征
    输入：
        features: 特征字典, 包含所有输入特征
        feat_type: 特征类型, 如"campaign_id"或"vmid"
        delta_type_feat: delta特征配置字典, 键为基础特征名, 值为时间窗口列表
        feat_bucket: 特征分桶配置字典, 用于验证生成的特征是否需要分桶
    输出：
        更新后的特征字典, 包含新生成的delta特征
    """
    # 参数初始化
    if not delta_type_feat:
        delta_type_feat = dict()
    if not feat_bucket:
        feat_bucket = dict()
    
    # 遍历每个基础特征及其时间窗口配置
    for k, v in delta_type_feat.items():
        pre_scale = v[0]  # 前一个时间窗口
        for scale in v[1:]:  # 当前时间窗口
            # 构建前一个时间窗口的特征名
            feat_pre = "{}_{}h_{}_all".format(k, pre_scale, feat_type)
            # 构建当前时间窗口的特征名
            feat_cur = "{}_{}h_{}_all".format(k, scale, feat_type)
            # 构建新的delta特征名
            feat_next = "{}_{}_{}h_{}_all".format(k, pre_scale, scale, feat_type)
            
            # 检查特征是否存在
            if feat_pre not in features:
                # print("feat_pre: {} not in feature!".format(feat_pre))
                continue
            elif feat_cur not in features:
                # print("feat_cur: {} not in feature!".format(feat_cur))
                continue
            
            # 检查新特征是否在分桶配置中
            if feat_next not in feat_bucket:
                print("feat_next: {} not in bucket".format(feat_next))
                continue
            else:
                print("feat_next: {} found in bucket".format(feat_next))
            
            # 计算delta特征值, 无效值(-1)的结果也设为-1
            features[feat_next] = tf.where(
                tf.logical_or(
                    tf.less_equal(features[feat_pre], tf.ones_like(features[feat_pre]) * -1),
                    tf.less_equal(features[feat_cur], tf.ones_like(features[feat_cur]) * -1) # bug修正：feat_pre-->feat_cur
                ),
                tf.ones_like(features[feat_pre], tf.float32) * -1,
                # 使用feat_delta函数计算两个特征的差值
                feat_delta(
                    tf.cast(features[feat_pre], tf.float32) / pre_scale,  # 标准化前一个特征
                    tf.cast(features[feat_cur], tf.float32) / scale  # 标准化当前特征
                )
            )
            pre_scale = scale  # 更新前一个时间窗口为当前时间窗口
    
    return features

def get_feat_type(feat_name, campaign_feat=None, campaign_vmid_feat=None):
    """
    特征类型判断函数
    功能：
        根据特征名称的后缀判断特征类型
    输入：
        feat_name: 特征名称字符串
        campaign_feat: 广告活动特征列表, 默认为None
        campaign_vmid_feat: 广告活动与素材交叉特征列表, 默认为None
    输出：
        特征类型字符串, 可能的值为："camp_x_vmid"、"vmid"、"camp"或"other"
    """
    # 判断是否为素材相关特征
    if feat_name.endswith('_vmid_all'):
        # 进一步判断是否为广告活动与素材的交叉特征
        if feat_name.endswith('_campaign_id_vmid_all'):
            return "camp_x_vmid"
        else:
            return "vmid"
    # 判断是否为广告活动特征
    elif feat_name.endswith('_campaign_id_all'):
        return "camp"
    elif feat_name.endswith('_camp'):
        return "camp"
    # 判断是否为广告活动与素材的交叉特征
    elif feat_name.endswith('_camp_vmid'):
        return "camp_x_vmid"
    # 其他类型特征
    else:
        # raise ValueError('unk featname: {}'.format(feat_name))
        # print("other featname: {}".format(feat_name))
        return "other"

def bucket_feats_2d(features, feat_to_bucket, log_dict=None, dim=1, need_log1p=True, need_reduce="", all_feat_suffix="",
                    need_summary=True, seq_pooling=False, pad_len=20, ratio_feat_list=None, ratio_feat=None,
                    delta_camp=None, delta_camp_vmid=None, delta_vmid=None, delta_feat=None, din=False,
                    save_camp_vmid_hourly=None):
    """
    二维特征的嵌入处理函数
    功能：
        处理二维特征, 包括ratio特征计算、delta特征处理、特征压缩和分桶嵌入等操作
    输入：
        features: 特征字典, 包含所有输入特征
        feat_to_bucket: 特征分桶点字典, 键为特征名称, 值为分桶边界点
        log_dict: 需要进行日志压缩的特征字典, 默认为None
        dim: 嵌入向量维度, 默认为1
        need_log1p: 是否需要进行log1p压缩, 默认为True
        need_reduce: 特征降维方式, 可选"max"或"mean", 默认为空字符串
        all_feat_suffix: 所有特征的后缀, 默认为空字符串
        need_summary: 是否需要记录特征摘要, 默认为True
        seq_pooling: 是否需要序列池化, 默认为False
        pad_len: 序列填充长度, 默认为20
        ratio_feat_list: 比率特征列表, 默认为None
        ratio_feat: 比率特征分桶点字典, 默认为None
        delta_camp: 广告活动delta特征字典, 默认为None
        delta_camp_vmid: 广告活动与素材交叉delta特征字典, 默认为None
        delta_vmid: 素材delta特征字典, 默认为None
        delta_feat: delta特征分桶点字典, 默认为None
        din: 是否使用DIN (Deep Interest Network) 模型, 默认为False
        save_camp_vmid_hourly: 存储每小时广告活动与素材特征的列表, 默认为None
    输出：
        state_embeddings: 嵌入向量列表
        state_embeddings_names: 对应嵌入向量的特征名称列表
        all_emb_size: 所有嵌入向量的总维度
        campaign_embeddings: 广告活动嵌入向量列表
        campaign_vmid_emb: 广告活动与素材交叉嵌入向量列表
        vmid_emb_size: 素材嵌入向量的总维度
    """

    # 字典初始化
    if not log_dict:
        log_dict = dict()
    if not ratio_feat_list:
        ratio_feat_list = list()
    if not ratio_feat:
        ratio_feat = dict()
    # 如果没有指定特征后缀, 则启用摘要记录
    if not all_feat_suffix:
        need_summary = True
    if not delta_camp:
        delta_camp = dict()
    if not delta_camp_vmid:
        delta_camp_vmid = dict()
    if not delta_vmid:
        delta_vmid = dict()
    if not delta_feat:
        delta_feat = dict()
    
    # 列表初始化
    if not save_camp_vmid_hourly:
        save_camp_vmid_hourly = list()
    state_embeddings = list()  # 存储生成的嵌入向量
    state_embeddings_names = list()  # 存储嵌入向量对应的特征名称
    multi_vmid_embeddings = list()  # 存储多个素材嵌入
    multi_vmid_embeddings_names = list()  # 存储多个素材嵌入对应的特征名称
    valid_masks = list()  # 存储有效掩码
    campaign_embeddings = list()  # 存储广告活动嵌入
    campaign_vmid_emb = list()  # 存储广告活动与素材交叉嵌入

    # 特征长度收集
    all_emb_size = 0
    vmid_emb_size = 0
    
    # ratio特征处理
    for feat_info in ratio_feat_list:
        # 跳过广告活动与素材交叉特征和广告活动特征
        if get_feat_type(feat_info.name_1) == 'camp_x_vmid' or get_feat_type(feat_info.name_1) == 'camp':
            print("skip invalid ratio feat_name: {}".format(feat_info.name_1))
            continue
        if get_feat_type(feat_info.name_2) == 'camp_x_vmid' or get_feat_type(feat_info.name_2) == 'camp':
            print("skip invalid ratio feat_name: {}".format(feat_info.name_2))
            continue
        # 跳过不存在的特征
        if feat_info.name_1 not in features or feat_info.name_2 not in features:
            continue
        # 再次检查特征类型
        if get_feat_type(feat_info.name_1) == 'camp_x_vmid' or get_feat_type(feat_info.name_2) == 'camp_x_vmid':
            print("skip invalid feat_name: {}".format(feat_info.name_1 if get_feat_type(feat_info.name_1) == 'camp_x_vmid' else feat_info.name_2))
            continue
        
        # 计算比率特征值, 对于无效值(-1), 结果也设为-1
        ratio_val = tf.where(
            tf.logical_or(
                tf.less_equal(features[feat_info.name_1], tf.ones_like(features[feat_info.name_1],
                                                                       features[feat_info.name_1].dtype) * -1),
                tf.less_equal(features[feat_info.name_2], tf.ones_like(features[feat_info.name_2],
                                                                       features[feat_info.name_2].dtype) * -1),
            ),
            tf.ones_like(features[feat_info.name_1], tf.float32) * -1,
            data_compress(features[feat_info.name_1]) - data_compress(
                features[feat_info.name_2])  # 计算压缩后的特征差值
        )
        # 将新生成的比率特征添加到特征字典
        features[feat_info.fc_name] = ratio_val
        # 将新生成的比率特征的分桶边界添加到分桶字典
        feat_to_bucket[feat_info.fc_name] = ratio_feat[feat_info.fc_name]

    # delta特征处理, 生成并收集新增delta特征
    # features = gen_delta_feat(features, feat_type="campaign_id", delta_type_feat=delta_camp, feat_bucket=delta_feat)
    features = gen_delta_feat(features, feat_type="vmid", delta_type_feat=delta_vmid, feat_bucket=delta_feat)
    
    # 收集新增delta特征的分桶边界
    for feat, info in delta_feat.items():
        # 跳过不存在的delta特征
        if feat not in features:
            print("skip invalid delta feat_name: {}".format(feat))
            continue
        print("valid delta feat_name: {}".format(feat))
        feat_to_bucket[feat] = info

    # 处理所有特征, 生成嵌入向量
    for feat_name, bucket_points in sorted(feat_to_bucket.items()):
        # 跳过广告活动与素材交叉特征和广告活动特征
        if get_feat_type(feat_name) == 'camp_x_vmid' or get_feat_type(feat_name) == 'camp':
            print("skip invalid dense_features_2d feat_name: {}".format(feat_name))
            continue
        
        # 获取特征值
        feat_val = features[feat_name]

        # 对于不是ratio、delta且未在log_dict中的特征, 进行数据压缩
        if need_log1p and feat_name not in ratio_feat and feat_name not in delta_feat: 
            feat_val = data_compress(feat_val)
        else:
            feat_val = tf.cast(features[feat_name], tf.float32)

        # 对特征进行降维处理
        if need_reduce == "max":
            feat_val = tf.reduce_max(feat_val, axis=-1, keepdims=True)  # 取最大值
        elif need_reduce == "mean":
            feat_val = tf.reduce_mean(feat_val, axis=-1, keepdims=True)  # 取平均值
        else:
            raise ValueError("unk need_reduce: {}".format(need_reduce))

        # 如果需要记录摘要, 记录特征的均值
        if need_summary:
            print('[bucket_feats_2d] lzx_debug:', feat_name, get_feat_type(feat_name), feat_val.get_shape())
            tf.summary.scalar('raw_feat_mean_{}_{}'.format(feat_name, all_feat_suffix), tf.reduce_mean(feat_val))

        # 对于在log_dict中的特征, 进行数据压缩并使用对应的分桶边界
        if feat_name in log_dict:
            feat_val = data_compress(feat_val)
            bucket_points = log_dict[feat_name]
        
        # 对特征进行分桶并生成嵌入向量
        emb_tensor = bucket_single_feat(feat_val, bucket_points, fc=feat_name, dim=dim, suffix=all_feat_suffix)
        state_embeddings.append(emb_tensor)
        state_embeddings_names.append(feat_name)

    print("len of dense_features_2d state_embeddings: {}".format(state_embeddings))
    print("dense_features_2d state_embeddings_names: {}".format(state_embeddings_names))
    return state_embeddings, state_embeddings_names, all_emb_size, campaign_embeddings, campaign_vmid_emb, vmid_emb_size

def bucket_feats(features, dict, log_dict=None, dict_v2=None, need_log1p=True, dim=1, lhuc_feat_names=None,
                 context_feat_names=None, filter_shadow=False, all_feat_suffix="", feat_2d=None):
    """
    一维特征的分桶和嵌入处理函数
    功能：
        对输入的一维密集特征进行分桶处理, 并生成对应的嵌入向量, 支持LHUC特征和上下文特征的特殊处理
    输入：
        features: 特征字典, 包含所有输入特征
        dict: 特征分桶点字典, 键为特征名称, 值为分桶边界点
        log_dict: 需要进行日志压缩的特征字典, 默认为None
        dict_v2: 备用特征分桶点字典, 默认为None
        need_log1p: 是否需要进行log1p压缩, 默认为True
        dim: 嵌入向量维度, 默认为1
        lhuc_feat_names: LHUC (Learning Hidden Unit Contributions) 特征名称集合, 默认为None
        context_feat_names: 上下文特征名称集合, 默认为None
        filter_shadow: 是否过滤包含"shadow"的特征, 默认为False
        all_feat_suffix: 所有特征的后缀, 默认为空字符串
        feat_2d: 二维特征列表, 默认为None
    输出：
        state_embeddings: 嵌入向量列表
        state_embeddings_names: 对应嵌入向量的特征名称列表
        lhuc_embeddings: LHUC特征的嵌入向量列表
        context_embeddings: 上下文特征的嵌入向量列表
        filtered_shadow_feats: 被过滤掉的shadow特征列表
    """
    # 参数初始化
    if log_dict is None:
        log_dict = {}
    if dict_v2 is None:
        dict_v2 = {}
    if lhuc_feat_names is None:
        lhuc_feat_names = set()
    if context_feat_names is None:
        context_feat_names = set()
    if feat_2d is None:
        feat_2d = list()
    
    # 结果列表初始化
    state_embeddings = list()  # 存储生成的嵌入向量
    state_embeddings_names = list()  # 存储嵌入向量对应的特征名称
    lhuc_embeddings = list()  # 存储LHUC特征的嵌入向量
    lhuc_embeddings_names = list()  # 存储LHUC特征的嵌入向量对应的特征名称
    context_embeddings = list()  # 存储上下文特征的嵌入向量
    context_embeddings_names = list()  # 存储上下文特征的嵌入向量对应的特征名称
    filtered_shadow_feats = list()  # 存储被过滤掉的shadow特征
    
    # 处理每个特征
    for fc, bucket_points in sorted(dict.items()):
        # 跳过广告活动与素材交叉特征和广告活动特征
        if get_feat_type(fc) == 'camp_x_vmid' or get_feat_type(fc) == 'camp':
            print("skip invalid dense_features_1d feat_name: {}".format(fc))
            continue
        # 如果需要过滤shadow特征且当前特征包含"shadow"
        if filter_shadow and "shadow" in fc:
            filtered_shadow_feats.append(fc)
            continue
        # 对于二维特征, 取最大值降维
        if fc in feat_2d:
            features[fc] = tf.reduce_max(features[fc], axis=-1, keepdims=True)
        # 获取特征值, 根据need_log1p决定是否进行数据压缩
        if need_log1p:  # need_log1p有log_dict时一般设置为False
            feat_val = data_compress(features[fc][:, 0])
        else:
            feat_val = tf.cast(features[fc], tf.float32)

        # 对于在log_dict中的特征, 进行数据压缩并使用对应的分桶边界
        if fc in log_dict:
            feat_val = data_compress(features[fc][:, 0])
            bucket_points = log_dict[fc]

        # 记录特征均值的平方根到摘要
        tf.summary.scalar('raw_feat_mean_' + fc,
                          tf.sqrt(tf.reduce_mean(feat_val)))
        
        # 如果是上下文特征, 单独处理
        if fc in context_feat_names:
            context_embeddings.append(bucket_single_feat(feat_val, bucket_points, fc=fc, dim=dim, prefix='context_',
                                                         suffix=all_feat_suffix))
            context_embeddings_names.append(fc)
            # context 特征不放在主要的塔里面
            continue
        
        # 生成并添加特征嵌入向量
        state_embeddings.append(
            bucket_single_feat(feat_val, bucket_points, fc=fc, dim=dim, suffix=all_feat_suffix)
        )
        state_embeddings_names.append(fc)
        
        # 如果是LHUC特征, 额外生成并添加LHUC嵌入向量
        if fc in lhuc_feat_names:
            lhuc_embeddings.append(
                bucket_single_feat(feat_val, bucket_points, fc=fc, dim=dim, prefix="lhuc", suffix=all_feat_suffix)
            )
            lhuc_embeddings_names.append(fc)
    
    print("len of dense_features_1d state_embeddings: {}".format(state_embeddings))
    print("dense_features_1d state_embeddings_names: {}".format(state_embeddings_names))
    return state_embeddings, state_embeddings_names, lhuc_embeddings, context_embeddings, filtered_shadow_feats

def get_shadow_number(features,
                  feat_name='fc_dense_ad_cid_shadow_num_max_strategy',
                  last_feat_name='fc_dense_ad_cid_shadow_num_max_strategy_last'):
    """
    获取广告id跑量期间前后两个时间步的shadow_num特征值
    功能：
        计算当前时间步和上一个时间步的shadow_num的平均值, 用于后续特征处理
    输入：
        features: 特征字典, 包含所有输入特征
        feat_name: 当前时间步的shadow_num特征名称, 默认为'fc_dense_ad_cid_shadow_num_max_strategy'
        last_feat_name: 上一个时间步的shadow_num特征名称, 默认为'fc_dense_ad_cid_shadow_num_max_strategy_last'
    输出：
        mean_shadow_num: 两个时间步shadow_num的平均值, 形状为[batch_size, 1]
    """
    # 获取当前时间步的shadow_num特征值并转换为float32类型
    cur_shadow_num = tf.cast(features[feat_name][:, 0], tf.float32)
    # 获取上一个时间步的shadow_num特征值并转换为float32类型
    last_shadow_num = tf.cast(features[last_feat_name][:, 0], tf.float32)
    # 计算两个时间步的平均值(注意这里除以0.2而不是2, 可能是特殊的数据处理需求)
    mean_shadow_num = (cur_shadow_num + last_shadow_num) / 0.2
    return mean_shadow_num

def get_shadow_ea(features, feat_name='fc_dense_external_action_last'):
    """
    获取shadow对应的外部动作(external action)特征值
    功能：
        从特征字典中提取指定的外部动作特征, 并转换为整型
    输入：
        features: 特征字典, 包含所有输入特征
        feat_name: 外部动作特征名称, 默认为'fc_dense_external_action_last'
    输出：
        转换为整型的外部动作特征值, 形状为[batch_size, 1]
    """
    # 从特征字典中获取指定特征, 并转换为int32类型
    return tf.cast(features[feat_name][:, 0], tf.int32)

def get_shadow_bucket_idx(features, feat_bucket_dict):
    """
    对特征字典中的张量进行分桶处理, 返回每个特征对应的桶号
    功能：
        遍历所有需要分桶的特征, 对每个特征值进行分桶, 并记录每个特征对应的桶号
    输入：
        features: 特征字典, 包含所有输入特征
        feat_bucket_dict: 特征分桶配置字典, 键为特征名称, 值为分桶边界点
    输出：
        ret: 字典, 键为特征名称, 值为对应的桶号张量, 形状为[batch_size, 1]
    """
    # 初始化结果字典
    ret = dict()
    # 遍历每个需要分桶的特征
    for fc, bucket_points in sorted(feat_bucket_dict.items()):
        # 获取特征值并转换为float32类型
        feat_val = tf.cast(features[fc], tf.float32)
        # 使用bucket_idx函数进行分桶
        bucket_info = bucket_idx(feat_val, bucket_points)
        # [batch, 1]
        ret[fc] = bucket_info
    return ret

def get_test_prefix(train_prefix, flag='bucket0'):
    """
    获取测试样本路径的前缀
    功能：
        从训练样本路径中提取测试样本路径的前缀部分
    输入：
        train_prefix: 训练样本路径前缀字符串
        flag: 用于定位路径分隔点的标志字符串, 默认为'bucket0'
    输出：
        测试样本路径前缀字符串
    """
    # 查找标志字符串在路径中的位置
    pos_prefix = train_prefix.find(flag)
    # 确保标志字符串存在于路径中
    assert (pos_prefix >= 0)
    # 返回标志字符串之前的部分作为测试路径前缀
    return train_prefix[:pos_prefix]

def scaled_tf_summary(name, tensor, mask):
    """
    计算带掩码的张量的平均值并记录到TensorBoard摘要
    功能：
        对输入张量应用掩码, 计算掩码区域的平均值, 并将结果记录到TensorBoard
    输入：
        name: 摘要名称
        tensor: 要计算的张量
        mask: 布尔类型的掩码张量, 与tensor形状兼容
    输出：
        无返回值, 但会将结果写入TensorBoard摘要
    """
    # 将张量和掩码展平, 应用掩码, 然后计算平均值并记录到摘要
    tf.summary.scalar(name, tf.reduce_mean(tf.boolean_mask(tf.reshape(tensor, [-1]), tf.reshape(mask, [-1]))))

def add_bias_output(features, feat_dict, bias_output, bucket_idxs, reward_predict,
                    reward_regression_loss, idea_regression_loss, reward, need_log1p=True):
    """
    分析不同分桶区间的bias输出和预测奖励之间的关系
    功能：
        对特征进行分桶, 计算每个分桶内的bias输出、预测奖励、真实奖励和损失值, 并记录相关指标
    输入：
        features: 特征字典, 包含所有输入特征
        feat_dict: 特征分桶配置字典, 键为特征名称, 值为分桶边界点
        bias_output: bias输出张量
        bucket_idxs: 需要分析的分桶索引列表
        reward_predict: 预测奖励张量
        reward_regression_loss: 奖励回归损失张量
        idea_regression_loss: 理想回归损失张量
        reward: 真实奖励张量
        need_log1p: 是否需要对特征进行log1p压缩, 默认为True
    输出：
        metric_tensors: 包含所有计算指标的字典
    """
    # 初始化指标字典
    metric_tensors = dict()
    # 遍历每个需要分桶的特征
    for fc, bucket_points in sorted(feat_dict.items()):
        # 根据need_log1p决定是否对特征进行log1p压缩
        if need_log1p:
            feat_val = data_compress(features[fc][:, 0])
        else:
            feat_val = tf.cast(features[fc], tf.float32)
        # 对特征进行分桶
        bucket_info = gen_math_ops.bucketize(
            input=feat_val, boundaries=bucket_points
        )
        # 遍历每个需要分析的分桶索引
        for bucket_idx in bucket_idxs:
            # 创建当前分桶的掩码
            bucket_mask_tensor = tf.equal(bucket_info, bucket_idx)
            bucket_mask_tensor = tf.reshape(bucket_mask_tensor, [-1])
            # 应用掩码获取当前分桶的各项值
            bias_output_gt = tf.boolean_mask(tf.reshape(bias_output, [-1]), bucket_mask_tensor)
            reward_gt = tf.boolean_mask(tf.reshape(reward, [-1]), bucket_mask_tensor)
            reward_predict_ea = tf.boolean_mask(tf.reshape(reward_predict, [-1]), bucket_mask_tensor)
            reward_regression_loss_ea = tf.boolean_mask(tf.reshape(reward_regression_loss, [-1]), bucket_mask_tensor)
            idea_regression_loss_ea = tf.boolean_mask(tf.reshape(idea_regression_loss, [-1]), bucket_mask_tensor)
            # 创建命名空间并记录各项指标到TensorBoard
            with tf.name_scope('bias_outputs_{}_{}'.format(fc, bucket_idx)):
                tf.summary.scalar('reward_predict_mean_{}_{}'.format(fc, bucket_idx),
                                  tf.reduce_mean(reward_predict_ea))
                tf.summary.scalar('rmse_loss_mean_{}_{}'.format(fc, bucket_idx),
                                  tf.reduce_mean(reward_regression_loss_ea))
                tf.summary.scalar('gt_mse_loss_mean_{}_{}'.format(fc, bucket_idx),
                                  tf.reduce_mean(idea_regression_loss_ea))
                tf.summary.scalar('avg_sample_ratio_{}_{}'.format(fc, bucket_idx),
                                  tf.reduce_mean(tf.cast(bucket_mask_tensor, tf.float32)))
                tf.summary.scalar('gt_reward_{}_{}'.format(fc, bucket_idx), tf.reduce_mean(reward_gt))
                tf.summary.scalar('bias_out_{}_{}'.format(fc, bucket_idx), tf.reduce_mean(bias_output_gt))
                # 更新指标字典
                metric_tensors.update({
                    'reward_predict_mean_{}_{}'.format(fc, bucket_idx): tf.reduce_mean(reward_predict_ea),
                    'rmse_loss_mean_{}_{}'.format(fc, bucket_idx): tf.reduce_mean(reward_regression_loss_ea),
                    'gt_mse_loss_mean_{}_{}'.format(fc, bucket_idx): tf.reduce_mean(idea_regression_loss_ea),
                    'avg_sample_ratio_{}_{}'.format(fc, bucket_idx): tf.reduce_mean(
                        tf.cast(bucket_mask_tensor, tf.float32)),
                    'gt_reward_{}_{}'.format(fc, bucket_idx): tf.reduce_mean(reward_gt),
                    'bias_out_{}_{}'.format(fc, bucket_idx): tf.reduce_mean(bias_output_gt)
                })
    return metric_tensors

def add_ea_metrics(stat_dense_ea_list, external_action_tensor, reward_predict,
                   reward_regression_loss, idea_regression_loss, reward):
    """
    分析不同外部动作(external action)类型的预测结果和性能指标
    功能：
        对不同外部动作类型的样本分别计算预测奖励、真实奖励和损失值, 并记录相关指标
    输入：
        stat_dense_ea_list: 需要分析的外部动作类型列表
        external_action_tensor: 外部动作张量
        reward_predict: 预测奖励张量
        reward_regression_loss: 奖励回归损失张量
        idea_regression_loss: 理想回归损失张量
        reward: 真实奖励张量
    输出：
        metric_tensors: 包含所有计算指标的字典
    """
    # 初始化指标字典
    metric_tensors = dict()
    # 遍历每个需要分析的外部动作类型
    for ea in stat_dense_ea_list:
        # 创建当前外部动作类型的掩码
        ea_mask_tensor = tf.equal(external_action_tensor, ea)
        ea_mask_tensor = tf.reshape(ea_mask_tensor, [-1])
        # 创建命名空间并记录各项指标到TensorBoard
        with tf.name_scope('predict_values_%d' % ea):
            # 应用掩码获取当前外部动作类型的各项值
            reward_gt = tf.boolean_mask(tf.reshape(reward, [-1]), ea_mask_tensor)
            reward_predict_ea = tf.boolean_mask(tf.reshape(reward_predict, [-1]), ea_mask_tensor)
            reward_regression_loss_ea = tf.boolean_mask(tf.reshape(reward_regression_loss, [-1]), ea_mask_tensor)
            idea_regression_loss_ea = tf.boolean_mask(tf.reshape(idea_regression_loss, [-1]), ea_mask_tensor)
            # 记录各项指标
            tf.summary.scalar('reward_predict_mean_{}'.format(ea), tf.reduce_mean(reward_predict_ea))
            tf.summary.scalar('rmse_loss_mean_{}'.format(ea), tf.reduce_mean(reward_regression_loss_ea))
            tf.summary.scalar('gt_mse_loss_mean_{}'.format(ea), tf.reduce_mean(idea_regression_loss_ea))
            tf.summary.scalar('avg_sample_ratio_{}'.format(ea), tf.reduce_mean(tf.cast(ea_mask_tensor, tf.float32)))
            tf.summary.scalar('gt_reward_{}'.format(ea), tf.reduce_mean(reward_gt))
            # 更新指标字典
            metric_tensors.update({
                'reward_predict_mean_{}'.format(ea): tf.reduce_mean(reward_predict_ea),
                'rmse_loss_mean_{}'.format(ea): tf.reduce_mean(reward_regression_loss_ea),
                'gt_mse_loss_mean_{}'.format(ea): tf.reduce_mean(idea_regression_loss_ea),
                'avg_sample_ratio_{}'.format(ea): tf.reduce_mean(tf.cast(ea_mask_tensor, tf.float32)),
                'gt_reward_{}'.format(ea): tf.reduce_mean(reward_gt)
            })
    return metric_tensors

def get_ea_list_mask(ea_tensor, stat_ea_list):
    """
    生成外部动作类型列表的掩码
    功能：
        生成一个布尔掩码, 标记哪些样本的外部动作类型属于指定的列表
    输入：
        ea_tensor: 外部动作张量, 形状为[batch_size, 1]
        stat_ea_list: 需要标记的外部动作类型列表, 如[1, 23, 3]
    输出：
        ea_mask_tensor: 布尔掩码张量, 形状为[batch_size, 1], 对应位置为True表示属于指定的动作类型列表
    """
    # 初始化掩码为全False
    ea_mask_tensor = tf.cast(tf.zeros_like(ea_tensor), tf.bool)
    # 遍历每个需要标记的外部动作类型
    for ea in stat_ea_list:
        # 逻辑或操作, 将当前外部动作类型的样本标记为True
        ea_mask_tensor = tf.math.logical_or(
            ea_mask_tensor, tf.equal(ea_tensor, ea)
        )
    return ea_mask_tensor

def get_dynamic_partition(ea_tensor, stat_ea_list, stat_ea_threshold):
    """
    根据外部动作类型动态划分样本并分配阈值
    功能：
        根据外部动作类型将样本划分为不同的组, 并为每个组分配对应的阈值
    输入：
        ea_tensor: 外部动作张量
        stat_ea_list: 外部动作类型列表, 元素可以是整数或元组
        stat_ea_threshold: 阈值列表, 长度为stat_ea_list长度+1
    输出：
        partitons: 分组结果张量, 形状与ea_tensor相同, 值表示所属的组索引
        thresholds: 阈值张量, 形状与ea_tensor相同, 值为对应组的阈值
    """
    # 确保阈值列表长度正确
    assert len(stat_ea_threshold) == len(stat_ea_list) + 1, "threshold: {} \t stat_ea_list: {}".format(
        len(stat_ea_threshold), len(stat_ea_list)
    )
    # 初始化分组结果为全0
    partitons = tf.zeros_like(ea_tensor)
    # 初始化阈值为第一个阈值
    thresholds = tf.ones_like(ea_tensor, tf.float32) * stat_ea_threshold[0]
    # 遍历每个外部动作类型
    for idx, eas in enumerate(stat_ea_list):
        # 处理元组类型的外部动作
        if isinstance(eas, tuple):
            # print ("test linhc: {}".format(eas))
            for ea in eas:
                # 更新属于当前外部动作的样本的分组和阈值
                partitons = tf.where(
                    tf.equal(ea_tensor, ea), tf.ones_like(ea_tensor) * (idx + 1), partitons)
                thresholds = tf.where(
                    tf.equal(ea_tensor, ea), tf.ones_like(ea_tensor, tf.float32) * stat_ea_threshold[idx + 1],
                    thresholds)
        # 处理整数类型的外部动作
        elif isinstance(eas, int):
            # 更新属于当前外部动作的样本的分组和阈值
            partitons = tf.where(
                tf.equal(ea_tensor, eas), tf.ones_like(ea_tensor) * (idx + 1), partitons)
            thresholds = tf.where(
                tf.equal(ea_tensor, eas), tf.ones_like(ea_tensor, tf.float32) * stat_ea_threshold[idx + 1], thresholds)
        # 处理不支持的类型
        else:
            raise ValueError("dynamic_partition error: {}".format(eas))
    return partitons, thresholds

def get_cost_level_partition(cost_tensor, cost_list, stat_threshold):
    """
    根据成本值动态划分样本并分配阈值
    功能：
        根据成本值将样本划分为不同的成本等级, 并为每个等级分配对应的阈值
    输入：
        cost_tensor: 成本值张量
        cost_list: 成本阈值列表, 如[25, 13]
        stat_threshold: 目标阈值列表, 长度为cost_list长度+1, 如[1.0, 1.x, 1.y]
    输出：
        partitons: 成本等级划分结果张量, 形状与cost_tensor相同, 值表示所属的成本等级
        thresholds: 阈值张量, 形状与cost_tensor相同, 值为对应成本等级的阈值
    """
    # 确保阈值列表长度正确
    assert len(stat_threshold) == len(cost_list) + 1, "stat_threshold: {} \t cost_list: {}".format(
        len(stat_threshold), len(cost_list)
    )
    # 初始化分组结果为全0
    partitons = tf.zeros_like(cost_tensor)
    # 初始化阈值为第一个阈值
    thresholds = tf.ones_like(cost_tensor, tf.float32) * stat_threshold[0]
    # 遍历每个成本阈值
    for idx, cost in enumerate(cost_list):
        # 处理浮点数类型的成本阈值
        if isinstance(cost, float):
            # 更新成本值小于等于当前阈值的样本的分组和阈值
            partitons = tf.where(
                tf.less_equal(cost_tensor, cost), tf.ones_like(cost_tensor) * (idx + 1), partitons)
            thresholds = tf.where(
                tf.less_equal(cost_tensor, cost), tf.ones_like(
                    cost_tensor, tf.float32) * stat_threshold[idx + 1], thresholds)
        # 处理不支持的类型
        else:
            raise ValueError("get_cost_level_partition error: {}".format(cost))
    return partitons, thresholds

def get_cost_level_partition_v2(cost_tensor, cost_list, treatment_tensor, treatment_list):
    """
    根据成本值和处理类型动态划分样本(v2版本)
    功能：
        根据成本值和处理类型的组合将样本划分为不同的组
    输入：
        cost_tensor: 成本值张量
        cost_list: 成本阈值列表, 如[25, 13]
        treatment_tensor: 处理类型张量, 如分为0和1
        treatment_list: 处理类型列表, 如[0, 1]
    输出：
        partitons: 分组结果张量, 形状与cost_tensor相同, 值表示所属的组索引
    """
    # 初始化分组结果为全0
    partitons = tf.zeros_like(cost_tensor)
    # 遍历每个成本阈值和处理类型的组合
    for idx, cost in enumerate(cost_list):
        for jdx, t in enumerate(treatment_list):
            # 确保成本值为浮点数, 处理类型为整数
            if isinstance(cost, float) and isinstance(t, int):
                # 更新同时满足成本阈值和处理类型条件的样本的分组
                partitons = tf.where(
                    tf.logical_and(
                        tf.less_equal(cost_tensor, cost),
                        tf.equal(treatment_tensor, t)
                    ), tf.ones_like(cost_tensor) * (
                        idx * len(treatment_list) + jdx
                    ), partitons)
            else:
                raise ValueError("get_cost_level_partition error: {}".format(cost))
    # 处理成本值大于所有阈值的样本
    for jdx, t in enumerate(treatment_list):
        if isinstance(t, int):
            # 更新成本值大于最大阈值且满足处理类型条件的样本的分组
            partitons = tf.where(
                tf.logical_and(
                    tf.greater(cost_tensor, cost_list[-1]),
                    tf.equal(treatment_tensor, t)
                ), tf.ones_like(
                    cost_tensor) * (len(cost_list) * len(treatment_list) + jdx), partitons)
    return partitons
    """
    return partions by ea_list
    Args:
        cost_tensor:
        cost_list:          [25,  13]
        treatment_tensor:    [分为0,  1]
        treatment_list:     [0, 1]

    Returns:
        partitons by cost_level: [0, 0, 1, 2, 1, 1, 0]

    """
    partitons = tf.zeros_like(cost_tensor)
    for idx, cost in enumerate(cost_list):
        for jdx, t in enumerate(treatment_list):
            if isinstance(cost, float) and isinstance(t, int):
                partitons = tf.where(
                    tf.logical_and(
                        tf.less_equal(cost_tensor, cost),
                        tf.equal(treatment_tensor, t)
                    ), tf.ones_like(cost_tensor) * (
                        idx * len(treatment_list) + jdx
                    ), partitons)
            else:
                raise ValueError("get_cost_level_partition error: {}".format(cost))
    for jdx, t in enumerate(treatment_list):
        if isinstance(t, int):
            partitons = tf.where(
                tf.logical_and(
                    tf.greater(cost_tensor, cost_list[-1]),
                    tf.equal(treatment_tensor, t)
                ), tf.ones_like(
                    cost_tensor) * (len(cost_list) * len(treatment_list) + jdx), partitons)
    return partitons

def senet(feat_emb, feat_cnt):
    """
    实现Squeeze-and-Excitation Networks (SENet)特征注意力机制
    功能：
        通过Squeeze和Excitation两个步骤, 自适应地重新校准各通道的特征响应, 增强有用特征, 抑制无用特征
    输入：
        feat_emb: 特征嵌入张量, 形状为[batch_size, feature_count, embedding_size]
        feat_cnt: 特征数量
    输出：
        weighted_feat_emb: 应用注意力权重后的特征嵌入张量, 形状与输入相同
        scaling_weight: 学习到的注意力权重张量, 形状为[batch_size, feature_count, 1]
    """
    # feat_emb shape (bs, f_cnt, emb_size)
    # Squeeze操作：对每个特征的嵌入向量计算平均值、最大值和总和
    feat_emb_mean = tf.reduce_mean(feat_emb, axis = -1)
    feat_emb_max = tf.reduce_max(feat_emb, axis = -1)
    feat_emb_sum = tf.reduce_sum(feat_emb, axis = -1)
    # 将三种聚合结果拼接起来, 形状变为[-1, slot_cnt * 3]
    squeeze_emb = tf.concat([feat_emb_mean, feat_emb_max, feat_emb_sum], axis = -1) # [-1, slot_cnt * 3]
    print("squeeze_emb: ", squeeze_emb.shape) #  (?, 1809)
    print("feat_cnt: ", feat_cnt)
    
    # Excitation操作：使用全连接层学习通道间的依赖关系
    # 第一个全连接层进行降维, 使用leaky_relu激活函数
    excitation_emb = tf.layers.dense(inputs=squeeze_emb, 
                                    units=feat_cnt//4,
                                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                    activation=tf.nn.leaky_relu,
                                    name='senet_sq')
    # 第二个全连接层进行升维, 恢复到原来的通道数
    excitation_emb = tf.layers.dense(inputs=excitation_emb, 
                                    units=feat_cnt,
                                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                    activation=tf.nn.leaky_relu,
                                    name='senet_ex')
    # 将输出通过sigmoid函数压缩到[0,1]范围, 并reshape为合适的形状作为注意力权重
    scaling_weight = tf.reshape(tf.nn.sigmoid(excitation_emb), [-1, feat_cnt, 1]) # [b, slot_cnt, 1]
    # 应用注意力权重到原始特征嵌入上
    weighted_feat_emb = scaling_weight * feat_emb 
    # weighted_feat_emb = weighted_feat_emb + feat_emb  # 可选的残差连接
    # print("scaling_weight: ", scaling_weight.shape)
    # print("feat_emb: ", feat_emb.shape)
    # print("weighted_feat_emb: ", weighted_feat_emb.shape)
    return weighted_feat_emb, scaling_weight

def get_dense_tower(dim_list, feat, name_scope, is_train, act_fun=tf.nn.leaky_relu, dropout_prob=0.0):
    """
    构建多层全连接网络塔(dense tower)
    功能：
        根据指定的维度列表构建一个全连接神经网络, 用于特征转换和表示学习
    输入：
        dim_list: 每层神经元数量的列表, 决定网络的结构
        feat: 输入特征张量
        name_scope: 命名空间前缀, 用于区分不同的网络组件
        is_train: 是否处于训练模式, 影响dropout层的行为
        act_fun: 隐藏层的激活函数, 默认为tf.nn.leaky_relu
        dropout_prob: dropout概率, 默认为0.0(不使用dropout)
    输出：
        feat: 网络的最终输出特征张量
    """
    # 遍历每一层的维度配置
    for i in range(len(dim_list)):
        # 构建全连接层, 最后一层使用sigmoid激活函数, 其他层使用指定的激活函数
        feat = tf.layers.dense(
            inputs=feat, units=dim_list[i],
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
            activation=(act_fun if i < len(dim_list) - 1 else tf.nn.sigmoid),
            name='{}{}'.format(name_scope, i))
        # 在训练模式下且非最后一层时应用dropout
        if is_train and dropout_prob > 0.0 and i < len(dim_list) - 1: # i < len(dim_list) - 1表示训练时最后一层不drop
            feat = tf.nn.dropout(feat, dropout_prob)
    return feat

def get_lhuc_tower(nn_input, nn_dims,
                   name, lhuc_input, first_dim=-1,
                   concat_nn_input=False, enable_bias=True,
                   lhuc_bottle_neck_dim=64, is_train=True, dropout_prob=0.0
                   ):
    """
    构建基于LHUC(Learning Hidden Unit Contributions)机制的特征塔
    功能：
        创建一个可调节的神经网络, 使用LHUC机制根据输入特征动态调整隐藏单元的贡献
    输入：
        nn_input: 主网络的输入特征张量
        nn_dims: 主网络各层的维度列表
        name: 网络名称前缀, 用于区分不同组件
        lhuc_input: LHUC机制的输入特征张量
        first_dim: 第一个LHUC层的维度, -1表示不使用自定义第一层
        concat_nn_input: 是否将主网络输入与LHUC输入拼接, 默认为False
        enable_bias: 是否启用LHUC偏置调整, 默认为True
        lhuc_bottle_neck_dim: LHUC网络中的瓶颈层维度, 默认为64
        is_train: 是否处于训练模式, 影响dropout行为
        dropout_prob: dropout概率, 默认为0.0
    输出：
        cur_layer: 应用LHUC机制后的网络输出张量
    """
    # 如果需要, 将主网络输入与LHUC输入拼接
    if concat_nn_input:
        lhuc_input = tf.concat([tf.stop_gradient(nn_input), lhuc_input], axis=1)
    # 确定LHUC网络的维度配置
    if first_dim <= 0:
        lhuc_dims = nn_dims[:-1]
    else:
        lhuc_dims = [first_dim] + nn_dims[:-1]
    
    # 初始化当前层为输入特征
    cur_layer = nn_input
    lhuc_idx = 0
    # 遍历主网络的每一层
    for i in range(len(nn_dims)):
        # 记录当前层的直方图用于监控
        tf.summary.histogram('{}_{}_cur_layer'.format(name, i), cur_layer)
        # 对隐藏层应用LHUC机制
        if i < len(nn_dims) - 1:
            if first_dim <= 0:
                # input不依赖lhuc
                continue
            lhuc_d = lhuc_dims[lhuc_idx]
            # print("lhuc_d", lhuc_d)
            
            # LHUC缩放因子计算：先通过瓶颈层降维
            lhuc_pre_scale = tf.layers.dense(
                inputs=lhuc_input,
                units=lhuc_bottle_neck_dim,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                activation=tf.nn.leaky_relu,
                name='{}_lhuc_inner_scale_{}'.format(name, i)
            )
            # 计算最终的缩放因子
            lhuc_scale = tf.layers.dense(
                inputs=lhuc_pre_scale,
                units=lhuc_d,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                activation=tf.nn.leaky_relu,
                name='{}_lhuc_scale_{}'.format(name, i)
            )
            # 更新LHUC维度索引
            lhuc_idx += 1
            # 使用tanh函数调整缩放范围, 并确保缩放因子大于0
            lhuc_scale = 1.0 + 3.0 * tf.nn.tanh(lhuc_scale)
            tf.summary.histogram('{}_{}_lhuc_scale'.format(name, i), lhuc_scale)
            # 应用缩放因子到当前层
            cur_layer = cur_layer * lhuc_scale
            
            # 如果启用偏置调整
            if enable_bias:
                # LHUC偏置计算：先通过瓶颈层降维
                lhuc_pre_bias = tf.layers.dense(
                    inputs=lhuc_input,
                    units=lhuc_bottle_neck_dim,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                    activation=tf.nn.leaky_relu,
                    name='{}_lhuc_pre_bias_{}'.format(name, i)
                )
                # 计算最终的偏置值
                lhuc_bias = tf.layers.dense(
                    inputs=lhuc_pre_bias,
                    units=lhuc_d,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                    activation=tf.nn.leaky_relu,
                    name='{}_lhuc_bias_{}'.format(name, i)
                )
                # 使用tanh函数限制偏置范围
                lhuc_bias = tf.nn.tanh(lhuc_bias)
                tf.summary.histogram('{}_{}_lhuc_bias'.format(name, i), lhuc_bias)
                # 应用偏置调整到当前层
                cur_layer = cur_layer + tf.nn.tanh(lhuc_bias)
        
        # 构建主网络的当前层
        cur_layer = tf.layers.dense(
            inputs=cur_layer, units=nn_dims[i],
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
            activation=(tf.nn.leaky_relu if i < len(nn_dims) - 1 else tf.nn.relu), 
            name='{}_main_{}'.format(name, i))
        
        # 在训练模式下且非最后一层时应用dropout
        if is_train and dropout_prob > 0.0 and i < len(nn_dims) - 1:
            cur_layer = tf.nn.dropout(cur_layer, dropout_prob)
    
    return cur_layer

def get_lhuc_out(state_tensor,nn_dims, name, lhuc_input, first_dim, concat_nn_input=False, enable_bias=True, lhuc_bottle_neck_dim=64, is_train=True, dropout_prob=0, task='cost'):
    """
    使用LHUC机制生成最终的输出预测
    功能：
        调用get_lhuc_tower获取LHUC特征塔的输出, 然后添加额外的偏置项并应用sigmoid激活得到最终预测
    输入：
        state_tensor: 状态张量, 作为LHUC塔的主输入
        nn_dims: LHUC塔各层的维度列表
        name: 网络名称前缀
        lhuc_input: LHUC机制的输入特征张量
        first_dim: 第一个LHUC层的维度
        concat_nn_input: 是否将主网络输入与LHUC输入拼接, 默认为False
        enable_bias: 是否启用LHUC偏置调整, 默认为True
        lhuc_bottle_neck_dim: LHUC网络中的瓶颈层维度, 默认为64
        is_train: 是否处于训练模式, 影响dropout行为
        dropout_prob: dropout概率, 默认为0
        task: 任务名称, 用于命名偏置层, 默认为'cost'
    输出：
        state_embedding_logit: 经过sigmoid激活的最终预测值, 范围在[0,1]之间
    """
    # 调用get_lhuc_tower获取LHUC特征塔的输出
    state_embedding_logit = get_lhuc_tower(
        state_tensor,
        nn_dims, name,
        lhuc_input, # 此时并非原始的adcnt表征, 而是融入了LHUC feature
        first_dim,
        concat_nn_input,
        enable_bias,
        lhuc_bottle_neck_dim, 
        is_train,
        dropout_prob
    )
    
    # 添加额外的偏置项：使用全连接层从LHUC输入生成偏置
    # bias项 nn option, 主成分计划数表征,  key-value memory network(part_1)
    logits_dense = tf.layers.dense(lhuc_input, 1, kernel_initializer=tf.glorot_normal_initializer(), activation=None, name="{}_layer_dense".format(task))
    
    # 结合LHUC塔输出和偏置项, 并应用sigmoid激活函数得到最终预测
    state_embedding_logit = tf.nn.sigmoid(state_embedding_logit + logits_dense) #最后一层激活函数用sigmoid(分类任务)
    
    return state_embedding_logit

def lhuc_net_scale_input(nn_input, nn_dims, name, lhuc_input,
                         first_dim, concat_nn_input=False, enable_bias=True, lhuc_bottle_neck_dim=64):
    """
    实现LHUC(Learning Hidden Unit Contributions)网络的输入缩放机制
    功能：
        通过LHUC机制根据输入特征动态调整神经网络各层的缩放因子和偏置
    输入：
        nn_input: 神经网络的输入特征张量
        nn_dims: 神经网络各层的维度列表
        name: 网络名称前缀
        lhuc_input: LHUC机制的输入特征张量
        first_dim: 第一个LHUC层的维度
        concat_nn_input: 是否将主网络输入与LHUC输入拼接, 默认为False
        enable_bias: 是否启用LHUC偏置调整, 默认为True
        lhuc_bottle_neck_dim: LHUC网络中的瓶颈层维度, 默认为64
    输出：
        cur_layer: 经过LHUC调整后的网络输出张量
    """
    # 如果需要, 将主网络输入与LHUC输入拼接
    if concat_nn_input:
        lhuc_input = tf.concat([tf.stop_gradient(nn_input), lhuc_input], axis=1)
    
    # 配置LHUC网络的维度, 包括自定义的第一层
    lhuc_dims = [first_dim] + nn_dims[:-1]
    
    # 初始化当前层为输入特征
    cur_layer = nn_input
    # 遍历网络的每一层
    for i in range(len(nn_dims)):
        # 记录当前层的直方图用于监控
        tf.summary.histogram('{}_{}_cur_layer'.format(name, i), cur_layer)
        lhuc_d = lhuc_dims[i]
        # print("lhuc_d", lhuc_d)
        
        # 对隐藏层应用LHUC机制
        if i < len(nn_dims) - 1:
            # LHUC缩放因子计算：先通过瓶颈层降维
            lhuc_pre_scale = tf.layers.dense(
                inputs=lhuc_input,
                units=lhuc_bottle_neck_dim,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                activation=tf.nn.leaky_relu,
                name='{}_lhuc_inner_scale_{}'.format(name, i)
            )
            # 计算最终的缩放因子
            lhuc_scale = tf.layers.dense(
                inputs=lhuc_pre_scale,
                units=lhuc_d,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                activation=tf.nn.leaky_relu,
                name='{}_lhuc_scale_{}'.format(name, i)
            )
            # 使用tanh函数调整缩放范围, 并确保缩放因子大于0
            lhuc_scale = 1.0 + 3.0 * tf.nn.tanh(lhuc_scale)
            tf.summary.histogram('{}_{}_lhuc_scale'.format(name, i), lhuc_scale)
            # 应用缩放因子到当前层
            cur_layer = cur_layer * lhuc_scale
            
            # 如果启用偏置调整
            if enable_bias:
                # LHUC偏置计算：先通过瓶颈层降维
                lhuc_pre_bias = tf.layers.dense(
                    inputs=lhuc_input,
                    units=lhuc_bottle_neck_dim,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                    activation=tf.nn.leaky_relu,
                    name='{}_lhuc_pre_bias_{}'.format(name, i)
                )
                # 计算最终的偏置值
                lhuc_bias = tf.layers.dense(
                    inputs=lhuc_pre_bias,
                    units=lhuc_d,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                    activation=tf.nn.leaky_relu,
                    name='{}_lhuc_bias_{}'.format(name, i)
                )
                # 使用tanh函数限制偏置范围
                lhuc_bias = tf.nn.tanh(lhuc_bias)
                tf.summary.histogram('{}_{}_lhuc_bias'.format(name, i), lhuc_bias)
                # 应用偏置调整到当前层
                cur_layer = cur_layer + tf.nn.tanh(lhuc_bias)
        
        # 构建神经网络的当前层
        cur_layer = tf.layers.dense(
            inputs=cur_layer, units=nn_dims[i],
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
            activation=(tf.nn.leaky_relu if i < len(nn_dims) - 1 else None),
            name='{}_main_{}'.format(name, i))
    
    return cur_layer

def feat_delta(feat_v1, feat_v2):
    """
    计算两个特征值之间的差值特征
    功能：
        根据两个特征值的大小关系, 计算对数差值特征, 确保输出值在合理范围内
    输入：
        feat_v1: 第一个特征值张量
        feat_v2: 第二个特征值张量
    输出：
        delta: 计算得到的差值特征张量
    """
    # 如果无效的处理
    # 根据feat_v1和feat_v2的大小关系计算对数差值
    # 注意：当feat_v2 <= feat_v1时, 由于使用了tf.nn.relu, 内部表达式结果为0, 导致整体结果为0或负数
    delta = tf.where(
            tf.less_equal(feat_v1, feat_v2), -1 * tf.math.log1p(
                tf.nn.relu(feat_v2 - feat_v1)),
            tf.math.log1p(
                tf.nn.relu(feat_v2 - feat_v1))
        ) # 疑问：此函数输出永远小于等于0, 符合预期吗？
    return delta 

def time_delta_feats(all_feats, feat_names, scales, name):
    """
    提取时间相关的差值特征
    功能：
        从多个时间点的特征中提取差值特征和比率特征, 用于捕获时间变化趋势
    输入：
        all_feats: 包含所有特征的字典
        feat_names: 时间点特征名称的列表, 如[1, 3, 7]表示不同时间点的信息
        scales: 对应每个特征的缩放因子列表
        name: 特征名称前缀, 用于生成新特征的名称
    输出：
        feat_dict: 包含生成的差值特征和比率特征的字典
    """
    # 确保特征名称列表和缩放因子列表长度一致
    assert len(feat_names) == len(scales), "feats: {} != scales:{}".format(len(feat_names), len(scales))
    
    # 初始化结果字典
    feat_dict = dict()
    
    # 遍历每对相邻的时间点特征
    for idx in range(0, len(feat_names) - 1):
        # 计算差值特征：使用feat_delta函数计算相邻时间点特征的差值
        feat_dict["delta_{}_{}".format(
            name, idx+1)] = feat_delta(
            tf.cast(all_feats[feat_names[idx]], tf.float32) / scales[idx],
            tf.cast(all_feats[feat_names[idx+1]], tf.float32) / scales[idx+1])
        
        # 计算比率特征：使用对数差表示比率关系
        feat_dict["ratio_{}_{}".format(
            name, idx+1)] = tf.math.log1p(
            tf.cast(all_feats[feat_names[idx]], tf.float32) / scales[idx]) - \
                          tf.math.log1p(tf.cast(all_feats[feat_names[idx]], tf.float32) / scales[idx+1])
    
    return feat_dict

def print_flags(flags):
    """
    打印所有配置参数(flags)的值
    功能：
        以格式化的方式打印所有配置参数及其当前值, 方便调试和日志记录
    输入：
        flags: 包含配置参数的对象, 具有__flags属性
    输出：
        无返回值, 但会将配置参数打印到控制台
    """
    # 打印开始标记
    print('#' * 80 + ' Print Flags Start ' + '#' * 80)
    # 遍历并打印每个参数及其值
    for flag, value in flags.__flags.items():
        print("Flag: %s, value: %s" % (flag, value.value))
    # 打印结束标记
    print('#' * 80 + ' Print Flags Over ' + '#' * 80)

if __name__ == "__main__":
    pass