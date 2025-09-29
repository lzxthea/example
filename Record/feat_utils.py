# encoding:utf-8

import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
import math


def data_compress(feat_val, need_compress=True):
    """
    数据压缩, 如果need_compress, log1p(x), 否则, 非负float32
    """
    if need_compress:
        return tf.math.log1p(tf.maximum(tf.cast(feat_val, tf.float32), 0.0))
    return tf.maximum(tf.cast(feat_val, tf.float32), 0.0)


def bucket_single_feat(feat_val, bucket_points, fc="fc", dim=1, prefix="", suffix="", need_summary=False):
    """
    输入的特征, embedding的维度
        feat_val: 单个特征值，形状通常为 [batch_size, 1]
        bucket_points: 分桶的边界点列表（如 [3, 5, 10])
        fc: 名称
        dim: 每个桶的嵌入向量长度（如 8 或 16)
        prefix: 名称前缀
        suffix: 名称后缀
        need_summary: 生成摘要信息（如桶分布直方图、均值），用于模型训练时的监控
    Returns:

    """
    # [batch, 1]
    bucket_num = len(bucket_points) + 1
    # [batch, 1]，分桶，每个元素是该样本对应的桶索引（如 0、1、2...）
    bucket_info = gen_math_ops.bucketize(
        input=feat_val, boundaries=bucket_points
    )
    if prefix:
        fc = prefix + fc
    if suffix:
        fc = fc + suffix
    if need_summary:
        # 用于训练时监控特征分桶的分布情况（如是否有桶过于稀疏或密集），帮助调整 bucket_points。
        tf.summary.histogram(fc + '_histogram', bucket_info)
        tf.summary.scalar(fc + 'bucket_mean', tf.reduce_mean(tf.cast(bucket_info, tf.float32)))
    # wv0 = tf.get_variable(name=fc + '_weights',
    #                       shape=[bucket_num, dim],
    #                       initializer=tf.random_uniform_initializer(-0.0078125, 0.0078125))
    # # 累计权重，非线性变换的同时具有线性含义，引入顺序性，如桶索引越大，嵌入向量单调递增
    # weights = tf.cumsum(tf.nn.softplus(wv0))
    # 生成可训练的嵌入权重矩阵：形状为 [bucket_num, dim]，即每个桶有一个 dim 维的向量，参数会在训练中学习。直接使用随机初始化的权重。
    weights = tf.get_variable(name=fc + '_weights',
                          shape=[bucket_num, dim],
                          initializer=tf.random_uniform_initializer(-0.0018125, 0.0018125))
    # [batch_size, 1, dim]，根据 bucket_info 中的桶索引，从 weights 中查询对应的嵌入向量
    emb_tensor = tf.gather(weights, bucket_info, axis=0)
    # 使用reshape
    emb_tensor = tf.reshape(emb_tensor, shape=[-1, dim])
    return emb_tensor

def _gen_center(buckets=None, bucket_lower_bound=0.0, bucket_upper_bound=70.1):
    """
    生成分桶的中心值
    Args:
        buckets: 分桶的边界点列表（如 [3, 5, 10]）
        bucket_lower_bound: 分桶的下界
        bucket_upper_bound: 分桶的上界
    Returns:
        分桶的中心值列表
    """
    if not buckets:
        buckets = [0.1, 2.1, 4.1, 6.1, 8.1, 10.1, 12.1, 14.1, 16.1, 18.1, 20.1, 22.1, 24.1, 26.1, 28.1, 30.1, 32.1, 34.1, 36.1, 38.1, 40.1, 42.1, 44.1, 46.1, 48.1, 50.1, 52.1, 54.1, 56.1, 58.1, 60.1]
    buckets = [bucket_lower_bound] + buckets + [bucket_upper_bound]
    mid_buckets = list()
    for i in range(0, len(buckets) - 1):
        mid_buckets.append((buckets[i] + buckets[i + 1]) / 2.0)
    return mid_buckets

# ad_cnt key-value mem network
def bucket_single_feat_semantic(feat_val, bucket_points, fc="fc",
                                dim=1, prefix="", suffix="", need_summary=False, method='auto',temp=0.8, bucket_lower_bound=0.0, bucket_upper_bound=70.1):
    """
    输入的特征, embedding的维度
        feat_val: 单个特征值，形状通常为 [batch_size, 1]
        bucket_points: 分桶的边界点列表（如 [3, 5, 10])
        fc: 名称
        dim: 每个桶的嵌入向量长度（如 8 或 16)
        prefix: 名称前缀
        suffix: 名称后缀
        need_summary: 生成摘要信息（如桶分布直方图、均值），用于模型训练时的监控
        method: 分桶方法，auto或manual
        temp: 温度参数，用于softmax归一化
        bucket_lower_bound: 分桶的下界
        bucket_upper_bound: 分桶的上界
    Returns:
        分桶后的嵌入向量

    """
    # [batch, 1]
    bucket_num = len(bucket_points) + 1
    # [batch, 1]
    bucket_info = gen_math_ops.bucketize(
        input=feat_val, boundaries=bucket_points
    )
    if prefix:
        fc = prefix + fc
    if suffix:
        fc = fc + suffix
    if need_summary:
        tf.summary.histogram(fc + '_histogram', bucket_info)
        tf.summary.scalar(fc + 'bucket_mean', tf.reduce_mean(tf.cast(bucket_info, tf.float32)))
    # wv0 = tf.get_variable(name=fc + '_weights',
    #                       shape=[bucket_num, dim],
    #                       initializer=tf.random_uniform_initializer(-0.0078125, 0.0078125))

    
    # # 累计权重，非线性变换的同时具有线性含义
    # weights = tf.cumsum(tf.nn.softplus(wv0))
    weights = tf.get_variable(name=fc + '_weights',
                              shape=[bucket_num, dim],
                              initializer=tf.random_uniform_initializer(-0.0018125, 0.0018125))
    
    # [batch, 1]，对特征值做log1p变换
    feat_val = tf.cast(feat_val, tf.float32)
    log_feat_val = tf.math.log1p(feat_val)
    # 定义第一层变换矩阵h_1（将log特征映射到桶维度）
    h_1 = tf.get_variable(name=fc + '_h1',
                          shape=[1, bucket_num],

                          initializer=tf.random_uniform_initializer(-0.0018125, 0.0018125))
    # 计算注意力权重 attn_w（形状 [batch_size, bucket_num]，表示每个样本对每个桶的关注程度）。
    # 通过神经网络自动学习注意力,两层神经网络 + 残差连接 + 温度缩放 + softmax 归一化
    # 1. 先压缩特征值范围（log1p），便于学习；
    # 2. 用第一层矩阵 h_1 得到初步关联度，加入非线性（leaky_relu）；
    # 3. 用第二层矩阵 w_1 深化关联，加入残差连接保留原始信息；
    # 4. 用温度参数调整分布陡峭程度，最后归一化得到权重。
    if method == 'auto':
        # raise ValueError("use auto method!")
        # [batch, bucket_num]
        # 用leaky_relu激活函数做非线性变换，得到特征与桶的初步关联
        log_feat_val_map = tf.nn.leaky_relu(tf.einsum(
            "ij,jk->ik", log_feat_val, h_1)
        )
        # 定义第二层变换矩阵w_1（进一步学习关联）
        w_1 = tf.get_variable(name=fc + '_W1',
                              shape=[bucket_num, bucket_num],
                              initializer=tf.random_uniform_initializer(-0.0018125, 0.0018125))
        # [batch, bucket_num] 计算注意力原始分数（attn_pre），加入原始特征映射作为残差
        attn_pre = tf.einsum(
            'ij,jk->ik', log_feat_val_map, w_1
        ) + 0.1 * log_feat_val_map
        t = 0.2
        #  用温度参数t调整分数分布（这里固定t=0.2，值越小权重越集中）
        attn_pre = attn_pre / t
        tf.summary.histogram(fc + '{}_attn_post'.format(method), attn_pre)
        # [batch, bucket_num],对调整后的分数进行softmax归一化，得到最终的注意力权重 attn_w
        attn_w = tf.nn.softmax(
            attn_pre, axis=-1
        )
    elif method == 'dis':
        # raise ValueError("use key value method!")
        mid_buckets = _gen_center(bucket_points, bucket_lower_bound=bucket_lower_bound, bucket_upper_bound=bucket_upper_bound)
        assert len(mid_buckets) == bucket_num, "{} != {}".format(len(mid_buckets), bucket_num)
        # [num_of_bucket]
        mid_constants = tf.reshape(tf.constant(mid_buckets, dtype=tf.float32), [1, len(mid_buckets)])
        feat_val = tf.reshape(feat_val, [-1, 1])
        
        # [batch, len(mid_buckets)]，将特征值扩展为与桶数量一致的形状（[batch_size, 1] → [batch_size, bucket_num]）
        feat_val = tf.tile(feat_val, [1, len(mid_buckets)])
        
        # [batch, len(mid_buckets)], 计算特征值与每个桶中心的绝对距离（距离越小，关联越强）
        dis_diff = tf.abs(feat_val - mid_constants)
        
        t = temp
        attn_pre = -1 * dis_diff / t
        tf.summary.histogram(fc + '{}_attn_post'.format(method), attn_pre)
        attn_w = tf.nn.softmax(attn_pre, axis=-1)
    else:
        raise ValueError('unk: {}'.format(method))
    # [batch, emb],矩阵乘法（爱因斯坦求和 einsum），用注意力权重 attn_w 对所有桶的嵌入向量 weights 进行加权求和，得到最终的特征嵌入（形状 [batch_size, dim]
    emb_tensor = tf.einsum('ij,jk->ik', attn_w, weights)
    return emb_tensor

def _debug_feat(feat, msg):
    """
    需要进行debug的特征, 打印：特征张量本身 + 特征的形状
    """
    feat = tf.Print(feat, [feat, tf.shape(feat)], message=msg, summarize=-1)
    return feat


def _pad_feat(features, fc, pad_feat):
    """
    features数据的 fc pad_feat, 处理特征字典中缺失的特征，通过填充默认值（零向量）确保特征的完整性
    """
    if fc not in features:
        return tf.zeros_like(pad_feat)
    return features[fc]


def bucket_idx(feat_val, bucket_points):
    """
    输入特征，计算出特征分桶之后的桶号
    Args:
        feat_val:
        bucket_points:

    Returns:
        tensor
    """
    bucket_info = gen_math_ops.bucketize(
        input=feat_val, boundaries=bucket_points
    )
    return bucket_info





def bucket_float_feats(features, cur_float_feat, feat_to_bucket, dim=1, filter_shadow=False):
    """
    pctr, pcvr特征
    Args:
        features:
        cur_float_feat:
        feat_to_bucket:
        dim:

    Returns:

    """
    state_embeddings = list()
    state_embeddings_names = list()
    filtered_shadow_feats = list()
    for feat_name in cur_float_feat:
        if filter_shadow and "shadow" in feat_name:
            filtered_shadow_feats.append(feat_name)
            continue
        feat_val = features[feat_name]
        tf.summary.scalar('raw_feat_mean_' + feat_name,
                          tf.sqrt(tf.reduce_mean(feat_val)))
        state_embeddings.append(
            bucket_single_feat(feat_val, feat_to_bucket[feat_name], fc=feat_name, dim=dim)
        )
        state_embeddings_names.append(feat_name)

    return state_embeddings, state_embeddings_names, filtered_shadow_feats


def _softmax_with_mask_(logits, masks, epsilon=1e-6):
    """
    Args:
    logits: [B, h, F, T]
    masks: [B, h, F, T]
    Return:
    [B, h, F, T]
    """
    max_logits = tf.reduce_max(logits, axis=-1, keepdims=True)
    safe_exp_logits = tf.exp(logits - max_logits)
    if masks is not None:
        safe_exp_logits = tf.where(masks, safe_exp_logits, tf.zeros_like(safe_exp_logits))

    return safe_exp_logits / (tf.reduce_sum(safe_exp_logits, axis=-1, keepdims=True) + epsilon)


def _attention_(queries, keys, values, masks=None, name_suffix=""):
    """
    Args:
    queries: [B, h, F, D]
    keys: [B, h, T, D]
    values: [B, h, T, D]
    masks: [B, h, T]
    name_suffix: summary的后缀信息
    Output:
    [B, h, F, D]
    """
    dim = queries.get_shape().as_list()[-1]
    attn_logits = tf.matmul(queries, keys, transpose_b=True) / math.sqrt(dim)  # [B, h, F, T]

    if masks is not None:
        from_seq_len = tf.shape(attn_logits)[-2]
        masks = tf.expand_dims(masks, axis=[-2])  # [B, h, 1, T]
        masks = tf.tile(masks, [1, 1, from_seq_len, 1])

    attn_weights = _softmax_with_mask_(attn_logits, masks)  # [B, h, F, T]

    valid_attn_w = tf.boolean_mask(
        attn_weights, masks
    )
    tf.summary.histogram(
        "valid_attn_w_{}".format(name_suffix), valid_attn_w
    )

    attn_output = tf.matmul(attn_weights, values)  # [B, h, F, D]

    return attn_output


def multi_head_attention(queries, keys, values, qk_dim, v_dim, head_num, masks=None, pad_len=20, name_suffix=""):
    """
    Args:
    queries: [B, F, H]
    keys: [B, T, H]
    values: [B, T, H]
    dim: int
    head_num: int
    masks: [B, T]
    pad_len:
    name_suffix:
    Output:
    [B, F, H]
    """

    def forward(x, layer_name, dim):
        dense_layer = tf.layers.dense(
            inputs=x, units=dim,
            use_bias=False,
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
            activation=None,
            name=layer_name)
        return dense_layer

    def split_head(x, dim, h_num, name=""):
        assert dim % h_num == 0, "dim: {}\t head_num: {}".format(dim, h_num)
        dim_per_head = int(dim / h_num)
        return tf.transpose(
            tf.reshape(x, [-1, pad_len, h_num, dim_per_head], name=name),
            perm=[0, 2, 1, 3])

    def merge_head(x, h_num, dim_per_head, name=""):
        return tf.reshape(
            tf.transpose(x, perm=[0, 2, 1, 3]),
            [-1, pad_len, h_num * dim_per_head], name=name)

    queries = forward(queries, "query_linear_" + name_suffix, qk_dim)  # [B, F, h*D]
    keys = forward(keys, "key_linear_" + name_suffix, qk_dim)  # [B, T, h*D]
    values = forward(values, "value_linear_" + name_suffix, v_dim)
    # values = forward(values, "value_linear")  # [B, T, h*D]

    queries = split_head(queries, qk_dim, head_num, "queries_" + name_suffix)  # [B, h, F, D1]
    keys = split_head(keys, qk_dim, head_num, "keys_" + name_suffix)  # [B, h, T, D1]
    values = split_head(values, v_dim, head_num, "values_" + name_suffix)  # [B, h, T, D2]

    if masks is not None:
        # Same mask applied to all h heads.
        masks = tf.expand_dims(masks, axis=[1])  # [B, 1, T]
        masks = tf.tile(masks, [1, head_num, 1])  # [B, h, T]

    attn_output = _attention_(queries, keys, values, masks, name_suffix=name_suffix)  # [B, h, F, D2]

    attn_output = merge_head(attn_output, head_num, int(v_dim / head_num), "attn_merge_" + name_suffix)  # [B, F, H * D2]

    return attn_output


def din_seq_model(query_emb, valid_masks, seq_embs, emb_size, name):
    """
    素材差异化建模
    Args:
        query_emb: [batch, emb_size]
        valid_masks: list_ele: [batch, pad_len] -> len of feat
        seq_embs: list_ele [batch, pad_len, emb_size] -> len of feat
        emb_size:
        name:

    Returns:

    """
    # 每个素材单独mask掉：
    # [b, num_of_feat, pad_len]
    assert len(valid_masks) == len(seq_embs), "len_of_masks: {} != seq_embs: {}".format(
        len(valid_masks), len(seq_embs)
    )
    # [b, pad_len, 1] len of feat
    valid_masks = [tf.cast(tf.expand_dims(m, axis=-1), tf.float32) for m in valid_masks]
    # [b, pad_len, len of feat]
    valid_mask_concat = tf.concat(valid_masks, axis=-1)
    # [b, pad_len]
    valid_num = tf.reduce_max(
        valid_mask_concat, axis=-1
    )
    # [b, 1]
    valid_num = tf.reduce_sum(
        valid_num, axis=1, keepdims=True
    )
    # valid_num = _debug_feat(valid_num, "linhc valid_num")
    # [b, pad_len, emb * len_of_feat]
    valid_mask_tensor = tf.concat(
        [tf.tile(m, [1, 1, emb_size]) for m in valid_masks], axis=-1
    )
    # print("name: {} valid_mask_tensor {} valid_num:{}".format(
    #     name, valid_mask_tensor.get_shape(), valid_num.get_shape()))
    tf.summary.histogram(
        "valid_num_{}".format(name),
        valid_num
    )
    # [b, pad_len, emb * num_of_feat]
    vmid_seq_embedding = tf.concat(
        seq_embs, axis=2
    )
    # [b, emb * num_of_feat]
    valid_num_seq = tf.reduce_sum(
        valid_mask_tensor, axis=1
    )
    # vmid_seq_embedding = _debug_feat(vmid_seq_embedding, "linhc vmid_seq_embedding")
    # valid_num_seq = _debug_feat(valid_num_seq, "linhc valid_num_seq")
    masked_vmid_seq = tf.reduce_sum(vmid_seq_embedding * valid_mask_tensor, axis=1)
    # [b, emb * num_of_feat]
    masked_vmid_seq = tf.math.divide_no_nan(masked_vmid_seq, valid_num_seq)
    # masked_vmid_seq = tf.where(
    #     tf.less_equal(valid_num_seq, tf.zeros_like(valid_num_seq)),
    #     tf.zeros_like(masked_vmid_seq),
    #     masked_vmid_seq / valid_num_seq
    # )
    return masked_vmid_seq, valid_num


def vmid_seq_model(multi_vmid_embeddings, valid_masks, qk_emb_size, emb_size, name, head_num=4, pad_len=20):
    """
    素材差异化建模
    Args:
        multi_vmid_embeddings: list_ele [batch, pad_len, emb_size]
        multi_vmid_embeddings_names:
        valid_masks: list_ele: [batch, pad_len]
        emb_size: 3 * num_of_feat
        name:
        head_num:
        pad_len:

    Returns:

    """
    # [b, num_of_feat, pad_len]
    valid_mask_tensor = tf.stack(
        valid_masks, axis=1
    )
    # print("name: {} vbefore alid_mask_tensor {}".format(
    #     name, valid_mask_tensor.get_shape()))
    # [b, pad_len]
    valid_mask_tensor = tf.reduce_max(
        tf.cast(valid_mask_tensor, tf.int32), axis=1
    )
    valid_num = tf.reduce_sum(
        tf.cast(valid_mask_tensor, tf.int32), axis=1, keepdims=True
    )
    # print("name: {} valid_mask_tensor {} valid_num:{}".format(
    #     name, valid_mask_tensor.get_shape(), valid_num.get_shape()))
    tf.summary.histogram(
        "valid_num_{}".format(name),
        valid_num
    )
    # [b, pad_len, emb * num_of_feat]
    vmid_seq_embedding = tf.concat(
        multi_vmid_embeddings, axis=2
    )
    valid_mask_tensor = tf.cast(valid_mask_tensor, tf.bool)
    atten_vmid_emb = multi_head_attention(
        vmid_seq_embedding, vmid_seq_embedding, vmid_seq_embedding,
        int(qk_emb_size), int(emb_size), int(head_num), masks=valid_mask_tensor, name_suffix=name, pad_len=pad_len)
    #[batch, seq_len, emb]
    #[batch, seq_len]
    atten_vmid_emb = tf.where(
        tf.tile(tf.expand_dims(valid_mask_tensor, axis=2), [1, 1, int(emb_size)]),
        atten_vmid_emb,
        tf.zeros_like(atten_vmid_emb)
    )
    # print("name: {} atten_vmid_emb {} valid_num:{}".format(
    #     name, atten_vmid_emb.get_shape(), valid_num.get_shape()))
    #[batch, seq_len]
    atten_vmid_emb = tf.reduce_sum(
        atten_vmid_emb, axis=1
    )
    # print("name: {} reduce_atten_vmid_emb {} valid_num:{}".format(
    #     name, atten_vmid_emb.get_shape(), valid_num.get_shape()))
    atten_vmid_emb = tf.where(
        tf.tile(valid_num > 0, [1, int(emb_size)]),
        atten_vmid_emb / tf.cast(valid_num, tf.float32),
        tf.zeros_like(atten_vmid_emb, tf.float32)
    )
    # print("name: {} last_atten_vmid_emb {} valid_num:{}".format(
    #     name, atten_vmid_emb.get_shape(), valid_num.get_shape()))
    return atten_vmid_emb, valid_num


def gen_delta_feat(features, feat_type="", delta_type_feat=None, feat_bucket=None):
    if not delta_type_feat:
        delta_type_feat = dict()
    if not feat_bucket:
        feat_bucket = dict()
    for k, v in delta_type_feat.items():
        pre_scale = v[0]
        for scale in v[1:]:
            feat_pre = "{}_{}h_{}_all".format(k, pre_scale, feat_type)
            feat_cur = "{}_{}h_{}_all".format(k, scale, feat_type)
            feat_next = "{}_{}_{}h_{}_all".format(k, pre_scale, scale, feat_type)
            if feat_pre not in features:
                # print("feat_pre: {} not in feature!".format(feat_pre))
                continue
            elif feat_cur not in features:
                # print("feat_cur: {} not in feature!".format(feat_cur))
                continue
            # assert feat_pre in features and feat_cur in features, 'feat_pre: {}\t feat_cur: {} not found!'.format(
            #     feat_pre, feat_cur)
            # assert feat_next not in features, 'feat_next: {} already exists!'.format(feat_next)
            if feat_next not in feat_bucket:
                # print("feat_next: {} not in bucket".format(feat_next))
                continue
            else:
                print("feat_next: {} found in bucket".format(feat_next))
            features[feat_next] = tf.where(
                tf.logical_or(
                    tf.less_equal(features[feat_pre], tf.ones_like(features[feat_pre]) * -1),
                    tf.less_equal(features[feat_cur], tf.ones_like(features[feat_cur]) * -1) # bug修正：feat_pre-->feat_cur
                ),
                tf.ones_like(features[feat_pre], tf.float32) * -1,
                feat_delta(
                    tf.cast(features[feat_pre], tf.float32) / pre_scale,
                    tf.cast(features[feat_cur], tf.float32) / scale
                )
            )
            pre_scale = scale
    return features


def get_feat_type(feat_name, campaign_feat=None, campaign_vmid_feat=None):
    if feat_name.endswith('_vmid_all'):
        if feat_name.endswith('_campaign_id_vmid_all'):
            return "camp_x_vmid"
        else:
            return "vmid"
    elif feat_name.endswith('_campaign_id_all'):
        return "camp"
    elif feat_name.endswith('_camp'):
        return "camp"
    elif feat_name.endswith('_camp_vmid'):
        return "camp_x_vmid"
    else:
        # raise ValueError('unk featname: {}'.format(feat_name))
        print("other featname: {}".format(feat_name))
        return "other"


def bucket_feats_2d(features, feat_to_bucket, log_dict=None, dim=1, need_log1p=True, need_reduce="", all_feat_suffix="",
                    need_summary=True, seq_pooling=False, pad_len=20, ratio_feat_list=None, ratio_feat=None,
                    delta_camp=None, delta_camp_vmid=None, delta_vmid=None, delta_feat=None, din=False,
                    save_camp_vmid_hourly=None):
    """
    2d特征的embedding
    Args:
        features:
        feat_to_bucket:
        dim:
        need_log1p:
        need_reduce:
        all_feat_suffix:
        need_summary:
        seq_pooling:
        pad_len: 20
    Returns:

    """

    # 字典初始化
    if not log_dict:
        log_dict = dict()
    if not ratio_feat_list:
        ratio_feat_list = list()
    if not ratio_feat:
        ratio_feat = dict()
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
    state_embeddings = list()
    state_embeddings_names = list()
    multi_vmid_embeddings = list()
    multi_vmid_embeddings_names = list()
    valid_masks = list()
    campaign_embeddings = list()
    campaign_vmid_emb = list()

    # 特征长度收集
    all_emb_size = 0
    vmid_emb_size = 0
    # ratio特征处理
    for feat_info in ratio_feat_list:
        if get_feat_type(feat_info.name_1) == 'camp_x_vmid' or get_feat_type(feat_info.name_1) == 'camp': # 跳过'camp'类型特征
            print("skip invalid ratio feat_name: {}".format(feat_info.name_1))
            continue
        if get_feat_type(feat_info.name_2) == 'camp_x_vmid' or get_feat_type(feat_info.name_2) == 'camp': # 跳过'camp'类型特征
            print("skip invalid ratio feat_name: {}".format(feat_info.name_2))
            continue
        if feat_info.name_1 not in features:
            continue
        if feat_info.name_2 not in features:
            continue
        if get_feat_type(feat_info.name_1) == 'camp_x_vmid': 
            print("skip invalid feat_name: {}".format(feat_info.name_1))
            continue
        if get_feat_type(feat_info.name_2) == 'camp_x_vmid':
            print("skip invalid feat_name: {}".format(feat_info.name_2))
            continue
        ratio_val = tf.where(
            tf.logical_or(
                tf.less_equal(features[feat_info.name_1], tf.ones_like(features[feat_info.name_1],
                                                                       features[feat_info.name_1].dtype) * -1),
                tf.less_equal(features[feat_info.name_2], tf.ones_like(features[feat_info.name_2],
                                                                       features[feat_info.name_2].dtype) * -1),
            ),
            tf.ones_like(features[feat_info.name_1], tf.float32) * -1,
            data_compress(features[feat_info.name_1]) - data_compress(
                features[feat_info.name_2])
        )
        # 收集新增ratio特征
        features[feat_info.fc_name] = ratio_val
        # 收集新增ratio特征的分桶边界
        feat_to_bucket[feat_info.fc_name] = ratio_feat[feat_info.fc_name]

    # delta特征处理，且收集新增delta特征
    # features = gen_delta_feat(features, feat_type="campaign_id", delta_type_feat=delta_camp, feat_bucket=delta_feat)
    features = gen_delta_feat(features, feat_type="vmid", delta_type_feat=delta_vmid, feat_bucket=delta_feat)
    # 收集新增ratio特征
    for feat, info in delta_feat.items():
        # 打印没找到的delta特征，且跳过
        if feat not in features:
            print(
                "skip invalid delta feat_name: {}".format(feat)
            )
            continue
        feat_to_bucket[feat] = info


    for feat_name, bucket_points in sorted(feat_to_bucket.items()):
        if get_feat_type(feat_name) == 'camp_x_vmid' or get_feat_type(feat_name) == 'camp': # 跳过'camp'类型特征
            print("skip invalid dense_features_2d feat_name: {}".format(feat_name))
            continue
        feat_val = features[feat_name]

        # ratio和delta以及log化过不再log化, need_log1p有log_dict时一般设置为False
        if need_log1p and feat_name not in ratio_feat and feat_name not in delta_feat: 
            feat_val = data_compress(feat_val)
        else:
            feat_val = tf.cast(features[feat_name], tf.float32)

        # reduce
        if need_reduce == "max":
            feat_val = tf.reduce_max(feat_val, axis=-1, keepdims=True)
        elif need_reduce == "mean":
            feat_val = tf.reduce_mean(feat_val, axis=-1, keepdims=True)
        else:
            raise ValueError("unk need_reduce: {}".format(need_reduce))

        if need_summary:
            print('[bucket_feats_2d] qhz_debug:', feat_name, get_feat_type(feat_name), feat_val.get_shape())
            tf.summary.scalar('raw_feat_mean_{}_{}'.format(feat_name, all_feat_suffix), tf.reduce_mean(feat_val))

        # 判断是否需要压缩：
        if feat_name in log_dict:
            feat_val = data_compress(feat_val)
            bucket_points = log_dict[feat_name]
        # bucket_single_feat
        emb_tensor = bucket_single_feat(feat_val, bucket_points, fc=feat_name, dim=dim, suffix=all_feat_suffix)
        state_embeddings.append(emb_tensor)
        state_embeddings_names.append(feat_name)

    print("len of dense_features_2d state_embeddings: {}".format(state_embeddings))
    print("dense_features_2d state_embeddings_names: {}".format(state_embeddings_names))
    return state_embeddings, state_embeddings_names, all_emb_size, campaign_embeddings, campaign_vmid_emb, vmid_emb_size




def bucket_feats(features, dict, log_dict=None, dict_v2=None, need_log1p=True, dim=1, lhuc_feat_names=None,
                 context_feat_names=None, filter_shadow=False, all_feat_suffix="", feat_2d=None):
    """
    对输入的dense特征进行hash分桶
    Args:
        features: [batch, num_of_feats]
        dict:
        log_dict:
        dict_v2:
        need_log1p:
        dim: 默认为1
        lhuc_feat_names:
        context_feat_names: aid下的mid相关特征，不作为serving阶段的信息输入源
        filter_shadow: 是否从输入中filter掉shadow相关的特征
        feat_2d: camp特征，用了20维的特征

    Returns:

    """
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
    state_embeddings = list()
    state_embeddings_names = list()
    lhuc_embeddings = list()
    lhuc_embeddings_names = list()
    context_embeddings = list()
    context_embeddings_names = list()
    filtered_shadow_feats = list()
    for fc, bucket_points in sorted(dict.items()):
        if get_feat_type(fc) == 'camp_x_vmid' or get_feat_type(fc) == 'camp': # 跳过'camp'类型特征
            print("skip invalid dense_features_1d feat_name: {}".format(fc))
            continue
        if fc in feat_2d:
            features[fc] = tf.reduce_max(features[fc], axis=-1, keepdims=True)
        if filter_shadow and "shadow" in fc:
            filtered_shadow_feats.append(fc)
            continue
        if need_log1p: # need_log1p有log_dict时一般设置为False
            feat_val = data_compress(features[fc][:, 0])
        else:
            feat_val = tf.cast(features[fc], tf.float32)

        # 使用log_dict
        if fc in log_dict:
            feat_val = data_compress(features[fc][:, 0])
            bucket_points = log_dict[fc]

        tf.summary.scalar('raw_feat_mean_' + fc,
                          tf.sqrt(tf.reduce_mean(feat_val)))
        if fc in context_feat_names:
            context_embeddings.append(bucket_single_feat(feat_val, bucket_points, fc=fc, dim=dim, prefix='context_',
                                                         suffix=all_feat_suffix))
            context_embeddings_names.append(fc)
            # context 特征不放在主要的塔里面
            continue
        state_embeddings.append(
            bucket_single_feat(feat_val, bucket_points, fc=fc, dim=dim, suffix=all_feat_suffix)
        )
        state_embeddings_names.append(fc)
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
    获取输入的aid跑量期间所有的前后两个time_step的shadow_num
    Returns:
    """
    cur_shadow_num = tf.cast(features[feat_name][:, 0], tf.float32)
    last_shadow_num = tf.cast(features[last_feat_name][:, 0], tf.float32)
    mean_shadow_num = (cur_shadow_num + last_shadow_num) / 0.2
    return mean_shadow_num


def get_shadow_ea(features, feat_name='fc_dense_external_action_last'):
    """
    获取shadow对应的ea
    Args:
        features:
        feat_name:

    Returns:

    """
    return tf.cast(features[feat_name][:, 0], tf.int32)


def get_shadow_bucket_idx(features, feat_bucket_dict):
    """
    对features中的tensor进行分桶，返回对应的桶号
    """
    ret = dict()
    for fc, bucket_points in sorted(feat_bucket_dict.items()):
        feat_val = tf.cast(features[fc], tf.float32)
        bucket_info = bucket_idx(feat_val, bucket_points)
        # [batch, 1]
        ret[fc] = bucket_info
    return ret


def get_test_prefix(train_prefix, flag='bucket0'):
    """
    获取test的样本路径前缀
    """
    pos_prefix = train_prefix.find(flag)
    assert (pos_prefix >= 0)
    return train_prefix[:pos_prefix]


def scaled_tf_summary(name, tensor, mask):
    tf.summary.scalar(name, tf.reduce_mean(tf.boolean_mask(tf.reshape(tensor, [-1]), tf.reshape(mask, [-1]))))


def add_bias_output(features, feat_dict, bias_output, bucket_idxs, reward_predict,
                    reward_regression_loss, idea_regression_loss, reward, need_log1p=True):
    """
    输出bias分数和 shadow_num之间的关系
    Returns:
    """
    metric_tensors = dict()
    for fc, bucket_points in sorted(feat_dict.items()):
        if need_log1p:
            feat_val = data_compress(features[fc][:, 0])
        else:
            feat_val = tf.cast(features[fc], tf.float32)
        bucket_info = gen_math_ops.bucketize(
            input=feat_val, boundaries=bucket_points
        )
        for bucket_idx in bucket_idxs:
            bucket_mask_tensor = tf.equal(bucket_info, bucket_idx)
            bucket_mask_tensor = tf.reshape(bucket_mask_tensor, [-1])
            bias_output_gt = tf.boolean_mask(tf.reshape(bias_output, [-1]), bucket_mask_tensor)
            reward_gt = tf.boolean_mask(tf.reshape(reward, [-1]), bucket_mask_tensor)
            reward_predict_ea = tf.boolean_mask(tf.reshape(reward_predict, [-1]), bucket_mask_tensor)
            reward_regression_loss_ea = tf.boolean_mask(tf.reshape(reward_regression_loss, [-1]), bucket_mask_tensor)
            idea_regression_loss_ea = tf.boolean_mask(tf.reshape(idea_regression_loss, [-1]), bucket_mask_tensor)
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
    分析不同ea的预测结果
    Returns:

    """
    metric_tensors = dict()
    for ea in stat_dense_ea_list:
        ea_mask_tensor = tf.equal(external_action_tensor, ea)
        ea_mask_tensor = tf.reshape(ea_mask_tensor, [-1])
        with tf.name_scope('predict_values_%d' % ea):
            reward_gt = tf.boolean_mask(tf.reshape(reward, [-1]), ea_mask_tensor)
            reward_predict_ea = tf.boolean_mask(tf.reshape(reward_predict, [-1]), ea_mask_tensor)
            reward_regression_loss_ea = tf.boolean_mask(tf.reshape(reward_regression_loss, [-1]), ea_mask_tensor)
            idea_regression_loss_ea = tf.boolean_mask(tf.reshape(idea_regression_loss, [-1]), ea_mask_tensor)
            tf.summary.scalar('reward_predict_mean_{}'.format(ea), tf.reduce_mean(reward_predict_ea))
            tf.summary.scalar('rmse_loss_mean_{}'.format(ea), tf.reduce_mean(reward_regression_loss_ea))
            tf.summary.scalar('gt_mse_loss_mean_{}'.format(ea), tf.reduce_mean(idea_regression_loss_ea))
            tf.summary.scalar('avg_sample_ratio_{}'.format(ea), tf.reduce_mean(tf.cast(ea_mask_tensor, tf.float32)))
            tf.summary.scalar('gt_reward_{}'.format(ea), tf.reduce_mean(reward_gt))
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
    return sample in stat_ea_list
    Args:
        ea_tensor: [batch, 1]
        stat_ea_list: [1, 23, 3]
    Returns:
        ea_mask_tensor: [batch, 1]  tf.bool

    """
    ea_mask_tensor = tf.cast(tf.zeros_like(ea_tensor), tf.bool)
    for ea in stat_ea_list:
        ea_mask_tensor = tf.math.logical_or(
            ea_mask_tensor, tf.equal(ea_tensor, ea)
        )
    return ea_mask_tensor


def get_dynamic_partition(ea_tensor, stat_ea_list, stat_ea_threshold):
    """
    return partions by ea_list
    Args:
        ea_tensor:
        stat_ea_list:
        stat_ea_threshold:
        default_threshold:

    Returns:
        partitons by eas: [0, 0, 1, 2, 1, 1, 0]

    """
    assert len(stat_ea_threshold) == len(stat_ea_list) + 1, "threshold: {} \t stat_ea_list: {}".format(
        len(stat_ea_threshold), len(stat_ea_list)
    )
    partitons = tf.zeros_like(ea_tensor)
    thresholds = tf.ones_like(ea_tensor, tf.float32) * stat_ea_threshold[0]
    for idx, eas in enumerate(stat_ea_list):
        if isinstance(eas, tuple):
            # print ("test linhc: {}".format(eas))
            for ea in eas:
                partitons = tf.where(
                    tf.equal(ea_tensor, ea), tf.ones_like(ea_tensor) * (idx + 1), partitons)
                thresholds = tf.where(
                    tf.equal(ea_tensor, ea), tf.ones_like(ea_tensor, tf.float32) * stat_ea_threshold[idx + 1],
                    thresholds)
        elif isinstance(eas, int):
            partitons = tf.where(
                tf.equal(ea_tensor, eas), tf.ones_like(ea_tensor) * (idx + 1), partitons)
            thresholds = tf.where(
                tf.equal(ea_tensor, eas), tf.ones_like(ea_tensor, tf.float32) * stat_ea_threshold[idx + 1], thresholds)
        else:
            raise ValueError("dynamic_partition error: {}".format(eas))
    return partitons, thresholds


def get_cost_level_partition(cost_tensor, cost_list, stat_threshold):
    """
    return partions by ea_list
    Args:
        cost_tensor:
        cost_list:          [25， 13]
        stat_threshold:     [1.0, 1.x, 1.y]

    Returns:
        partitons by cost_level: [0, 0, 1, 2, 1, 1, 0]

    """
    assert len(stat_threshold) == len(cost_list) + 1, "stat_threshold: {} \t cost_list: {}".format(
        len(stat_threshold), len(cost_list)
    )
    partitons = tf.zeros_like(cost_tensor)
    thresholds = tf.ones_like(cost_tensor, tf.float32) * stat_threshold[0]
    for idx, cost in enumerate(cost_list):
        if isinstance(cost, float):
            partitons = tf.where(
                tf.less_equal(cost_tensor, cost), tf.ones_like(cost_tensor) * (idx + 1), partitons)
            thresholds = tf.where(
                tf.less_equal(cost_tensor, cost), tf.ones_like(
                    cost_tensor, tf.float32) * stat_threshold[idx + 1], thresholds)
        else:
            raise ValueError("get_cost_level_partition error: {}".format(cost))
    return partitons, thresholds


def get_cost_level_partition_v2(cost_tensor, cost_list, treatment_tensor, treatment_list):
    """
    return partions by ea_list
    Args:
        cost_tensor:
        cost_list:          [25， 13]
        treatment_tensor:    [分为0， 1]
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

def get_dense_tower(dim_list, feat, name_scope, is_train, act_fun=tf.nn.leaky_relu, dropout_prob=0.0):
    """
    对其中的dense_tower的建模
    Args:
        dim_list:
        feat:
        name_scope:
        is_train:
        act_fun:
        dropout_prob:


    Returns:

    """
    for i in range(len(dim_list)):
        feat = tf.layers.dense(
            inputs=feat, units=dim_list[i],
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
            activation=(act_fun if i < len(dim_list) - 1 else tf.nn.sigmoid),
            name='{}{}'.format(name_scope, i))
        if is_train and dropout_prob > 0.0 and i < len(dim_list) - 1: # i < len(dim_list) - 1表示训练时最后一层不drop
            feat = tf.nn.dropout(feat, dropout_prob)
    return feat



def get_lhuc_tower(nn_input, nn_dims,
                   name, lhuc_input, first_dim=-1,
                   concat_nn_input=False, enable_bias=True,
                   lhuc_bottle_neck_dim=64, is_train=True, dropout_prob=0.0
                   ):
    """

    Args:
        nn_input:
        nn_dims:
        name:
        lhuc_input:
        first_dim:
        concat_nn_input:
        enable_bias:
        lhuc_bottle_neck_dim:

    Returns:

    """
    if concat_nn_input:
        lhuc_input = tf.concat([tf.stop_gradient(nn_input), lhuc_input], axis=1)
    if first_dim <= 0:
        lhuc_dims = nn_dims[:-1]
    else:
        lhuc_dims = [first_dim] + nn_dims[:-1]
    cur_layer = nn_input
    lhuc_idx = 0
    for i in range(len(nn_dims)):
        tf.summary.histogram('{}_{}_cur_layer'.format(name, i), cur_layer)
        if i < len(nn_dims) - 1:
            if first_dim <= 0:
                # input不依赖lhuc
                continue
            lhuc_d = lhuc_dims[lhuc_idx]
            # print("lhuc_d", lhuc_d)
            lhuc_pre_scale = tf.layers.dense(
                inputs=lhuc_input,
                units=lhuc_bottle_neck_dim,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                activation=tf.nn.leaky_relu,
                name='{}_lhuc_inner_scale_{}'.format(name, i)
            )
            lhuc_scale = tf.layers.dense(
                inputs=lhuc_pre_scale,
                units=lhuc_d,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                activation=tf.nn.leaky_relu,
                name='{}_lhuc_scale_{}'.format(name, i)
            )
            lhuc_idx += 1
            lhuc_scale = 1.0 + 3.0 * tf.nn.tanh(lhuc_scale)
            tf.summary.histogram('{}_{}_lhuc_scale'.format(name, i), lhuc_scale)
            cur_layer = cur_layer * lhuc_scale
            if enable_bias:
                lhuc_pre_bias = tf.layers.dense(
                    inputs=lhuc_input,
                    units=lhuc_bottle_neck_dim,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                    activation=tf.nn.leaky_relu,
                    name='{}_lhuc_pre_bias_{}'.format(name, i)
                )
                lhuc_bias = tf.layers.dense(
                    inputs=lhuc_pre_bias,
                    units=lhuc_d,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                    activation=tf.nn.leaky_relu,
                    name='{}_lhuc_bias_{}'.format(name, i)
                )
                lhuc_bias = tf.nn.tanh(lhuc_bias)
                tf.summary.histogram('{}_{}_lhuc_bias'.format(name, i), lhuc_bias)
                cur_layer = cur_layer + tf.nn.tanh(lhuc_bias)
        cur_layer = tf.layers.dense(
            inputs=cur_layer, units=nn_dims[i],
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
            activation=(tf.nn.leaky_relu if i < len(nn_dims) - 1 else tf.nn.relu), 
            name='{}_main_{}'.format(name, i))
        if is_train and dropout_prob > 0.0 and i < len(nn_dims) - 1:
            cur_layer = tf.nn.dropout(cur_layer, dropout_prob)
    return cur_layer


def get_lhuc_out(state_tensor,nn_dims, name, lhuc_input, first_dim, concat_nn_input=False, enable_bias=True, lhuc_bottle_neck_dim=64, is_train=True, dropout_prob=0, task='cost'):
    state_embedding_logit = get_lhuc_tower(
        state_tensor,
        nn_dims, name,
        lhuc_input, # 此时并非原始的adcnt表征，而是融入了LHUC feature
        first_dim,
        concat_nn_input,
        enable_bias,
        lhuc_bottle_neck_dim, 
        is_train,
        dropout_prob
    )
    # bias项 nn option, 主成分计划数表征， key-value memory network（part_1）
    logits_dense = tf.layers.dense(lhuc_input, 1, kernel_initializer=tf.glorot_normal_initializer(), activation=None, name="{}_layer_dense".format(task))
    state_embedding_logit = tf.nn.sigmoid(state_embedding_logit + logits_dense) #最后一层激活函数用sigmoid（分类任务）
    return state_embedding_logit


def lhuc_net_scale_input(nn_input, nn_dims, name, lhuc_input,
                         first_dim, concat_nn_input=False, enable_bias=True, lhuc_bottle_neck_dim=64):
    if concat_nn_input:
        lhuc_input = tf.concat([tf.stop_gradient(nn_input), lhuc_input], axis=1)
    lhuc_dims = [first_dim] + nn_dims[:-1]
    cur_layer = nn_input
    for i in range(len(nn_dims)):
        tf.summary.histogram('{}_{}_cur_layer'.format(name, i), cur_layer)
        lhuc_d = lhuc_dims[i]
        # print("lhuc_d", lhuc_d)
        if i < len(nn_dims) - 1:
            lhuc_pre_scale = tf.layers.dense(
                inputs=lhuc_input,
                units=lhuc_bottle_neck_dim,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                activation=tf.nn.leaky_relu,
                name='{}_lhuc_inner_scale_{}'.format(name, i)
            )
            lhuc_scale = tf.layers.dense(
                inputs=lhuc_pre_scale,
                units=lhuc_d,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                activation=tf.nn.leaky_relu,
                name='{}_lhuc_scale_{}'.format(name, i)
            )
            lhuc_scale = 1.0 + 3.0 * tf.nn.tanh(lhuc_scale)
            tf.summary.histogram('{}_{}_lhuc_scale'.format(name, i), lhuc_scale)
            cur_layer = cur_layer * lhuc_scale
            if enable_bias:
                lhuc_pre_bias = tf.layers.dense(
                    inputs=lhuc_input,
                    units=lhuc_bottle_neck_dim,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                    activation=tf.nn.leaky_relu,
                    name='{}_lhuc_pre_bias_{}'.format(name, i)
                )
                lhuc_bias = tf.layers.dense(
                    inputs=lhuc_pre_bias,
                    units=lhuc_d,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                    activation=tf.nn.leaky_relu,
                    name='{}_lhuc_bias_{}'.format(name, i)
                )
                lhuc_bias = tf.nn.tanh(lhuc_bias)
                tf.summary.histogram('{}_{}_lhuc_bias'.format(name, i), lhuc_bias)
                cur_layer = cur_layer + tf.nn.tanh(lhuc_bias)
        cur_layer = tf.layers.dense(
            inputs=cur_layer, units=nn_dims[i],
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
            activation=(tf.nn.leaky_relu if i < len(nn_dims) - 1 else None),
            name='{}_main_{}'.format(name, i))
    return cur_layer


def feat_delta(feat_v1, feat_v2):
    """

    Args:
        feat_v1:
        feat_v2:

    Returns:

    """
    # 如果无效的处理
    delta = tf.where(
            tf.less_equal(feat_v1, feat_v2), -1 * tf.math.log1p(
                tf.nn.relu(feat_v2 - feat_v1)),
            tf.math.log1p(
                tf.nn.relu(feat_v2 - feat_v1))
        ) # 疑问：此函数输出永远小于等于0，符合预期吗？
    return delta 


def time_delta_feats(all_feats, feat_names, scales, name):
    """
    提取出time delta 信息
    Args:
        all_feats:
        feat_names:  [1, 3, 7] info
        scales:
        name:

    Returns:

    """
    assert len(feat_names) == len(scales), "feats: {} != scales:{}".format(len(feat_names), len(scales))
    feat_dict = dict()
    for idx in range(0, len(feat_names) - 1):
        feat_dict["delta_{}_{}".format(
            name, idx+1)] = feat_delta(
            tf.cast(all_feats[feat_names[idx]], tf.float32) / scales[idx],
            tf.cast(all_feats[feat_names[idx+1]], tf.float32) / scales[idx+1])
        feat_dict["ratio_{}_{}".format(
            name, idx+1)] = tf.math.log1p(
            tf.cast(all_feats[feat_names[idx]], tf.float32) / scales[idx]) - \
                          tf.math.log1p(tf.cast(all_feats[feat_names[idx]], tf.float32) / scales[idx+1])
    return feat_dict


def print_flags(flags):
    print('#' * 80 + ' Print Flags Start ' + '#' * 80)
    for flag, value in flags.__flags.items():
        print("Flag: %s, value: %s" % (flag, value.value))
    print('#' * 80 + ' Print Flags Over ' + '#' * 80)


if __name__ == "__main__":
    pass