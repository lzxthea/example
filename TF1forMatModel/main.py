# encoding: utf-8
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf
import numpy as np  
np.set_printoptions(threshold=np.inf)  

from collections import defaultdict

from dataset import create_instance_dataset, CustomMetricHook

from feature_all import DENSE_FEAT_2D_more, SPARSE_FEAT_v2, \
    DENSE_FEAT_1D_v2, NEED_LOG_FEATURE_DICT, \
    RATIO_FEATURE, RATIO_FEATURE_INFO,  \
    DELTA_CAMP_FEAT, DELTA_CAMP_VMID_FEAT, DELTA_VMID_FEAT, DELTA_FEATURE_INFO

from utils.feat_utils import bucket_feats, get_dense_tower, bucket_feats_2d, get_lhuc_out, print_flags, bucket_single_feat_semantic, senet

from utils.metric_utils import get_AUC
from utils.loss_utils import get_vars_not_in_scope, get_pow_w, huber_loss, weighted_cross_entropy, cross_entropy, focal_loss, distill_softmax

from lagrange_lite import sparse
from lagrange_lite.tensorflow.train import DeepInsight2Hook, SynchronizedCheckpointSaverHook
from lagrange_lite.common import deep_insight_v2
from lagrange_lite.tensorflow import train
from lagrange_lite.common import JOB_CONTEXT
from lagrange_lite.tensorflow import aop
from datetime import datetime, timedelta
import warnings
# 设置忽略所有警告
warnings.filterwarnings('ignore')

FLAGS = tf.app.flags.FLAGS
# 训练参数
tf.app.flags.DEFINE_string('model_path', './', 'Model save path.')
tf.app.flags.DEFINE_string('train_paths', '', 'Train path')
tf.app.flags.DEFINE_string('test_paths', '', 'Test path')
tf.app.flags.DEFINE_string('last_model_path', '', 'last model path.')
tf.app.flags.DEFINE_integer('is_train', 1, 'train mode')
tf.app.flags.DEFINE_integer('is_eval', 0, 'eval mode')
tf.app.flags.DEFINE_integer('ps_num_embedding_shards', 1, 'ps number')
tf.app.flags.DEFINE_integer('batch_size', 512, 'Testing batch size.')
# tf.app.flags.DEFINE_integer('shuffle_buffer_size', 300000, 'Shuffle buffer size')
tf.app.flags.DEFINE_integer('shuffle_buffer_size', 8192, 'Shuffle buffer size')
tf.app.flags.DEFINE_integer('n_epochs', 1, 'Number of epochs.')
tf.app.flags.DEFINE_integer('save_checkpoints_secs', 300, 'Time interval for saving checkpoints.')
tf.app.flags.DEFINE_integer('save_summary_steps', 10, 'Sumarry steps.')
tf.app.flags.DEFINE_integer('log_step_count_steps', 100, 'Log steps.')
tf.app.flags.DEFINE_integer('keep_checkpoint_max', 10, 'Checkpoint maximum number.')
tf.app.flags.DEFINE_integer('cycle_length', 8, 'cycle_length')
tf.app.flags.DEFINE_integer('block_length', 2, 'block_length')
tf.app.flags.DEFINE_integer('num_parallel_maps', 256, 'num_parallel_maps')
tf.app.flags.DEFINE_bool('batch_reload', True, '天级别追新的时候是否reload模型')
tf.app.flags.DEFINE_float('deep_instance_sample_ratio', 0.1, 'deep_instance_sample_ratio')
tf.app.flags.DEFINE_float('huber_delta', 1.0, 'huber_delta')

# 模型参数
tf.app.flags.DEFINE_integer('seed', 9431, 'random seed')
tf.app.flags.DEFINE_string('dnn_hidden_dims_common', '[512,256,128,64]', 'DNN hidden dimensionality list in common.')
tf.app.flags.DEFINE_string('small_dnn_hidden_dims_common', '[128,64]', 'small DNN hidden dimensionality list in common.')
tf.app.flags.DEFINE_string('shared_dnn_hidden_dims_common', '[128,64]', 'SHARED DNN hidden dimensionality list in common.')
tf.app.flags.DEFINE_string('bias_hidden_dims_common', '[16, 4]', 'bias hidden dimensionality list 2 in common.')
tf.app.flags.DEFINE_string('need_task', 'convert,cost,convert_per_hour_label,cost_per_hour_label,convert_per_ad_hour_label,cost_per_ad_hour_label', '是否选择某个任务')
tf.app.flags.DEFINE_integer('sparse_emb_dim', '4', '稀疏特征的emb_size')
tf.app.flags.DEFINE_integer('sparse_cross_emb_dim', '2', '稀疏特征的emb_size')
tf.app.flags.DEFINE_integer('embedding_dim', '4', '参数embedding后的维度')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
tf.app.flags.DEFINE_float('ema_decay', 0.99, 'EMA decay for hooks.')
tf.app.flags.DEFINE_float('dropout_prob', 0.0, '是否使用dropout')
tf.app.flags.DEFINE_float('cost_thresh', 100.0, 'cost_level的切换阈值.')
tf.app.flags.DEFINE_bool('separate_embedding', True, '是否使用隔离embedding')
tf.app.flags.DEFINE_bool('separate_dense_embedding', False, '是否使用隔离dense embedding')
tf.app.flags.DEFINE_bool('seq_pooling', False, '是否使用序列感知建模')
tf.app.flags.DEFINE_bool('use_lhuc', False, '是否使用lhuc')
tf.app.flags.DEFINE_bool('use_recify', False, '是否把预估分数纠偏回来')
tf.app.flags.DEFINE_bool('transfer_learning', True, '是否迁移')
tf.app.flags.DEFINE_bool('use_log_feature', True, 'use log feature.')
tf.app.flags.DEFINE_bool('use_din', True, 'use_din')
tf.app.flags.DEFINE_bool('only_netservice', False, '是否只用网服样本')
tf.app.flags.DEFINE_integer('auto_type_to_keep', -1, '是否过滤')
tf.app.flags.DEFINE_bool('enable_ad_avg_uplift', True, 'enable_ad_avg_uplift')
tf.app.flags.DEFINE_float('label_cap_value_lb', 0.0, 'label_cap_value_lb')
tf.app.flags.DEFINE_float('label_cap_value_ub', 300.0, 'label_cap_value_ub') #label的上下界, 当前UBA在脚本里写死了300
tf.app.flags.DEFINE_string('loss_type', 'weighted_cross_entropy', 'loss_type')
tf.app.flags.DEFINE_float("temp", 10, "")
tf.app.flags.DEFINE_integer('adcnt_embedding_dim', 4, 'ad_cnt embedding后的维度')
tf.app.flags.DEFINE_integer('is_offline', 0, 'eval mode')
tf.app.flags.DEFINE_float('cross_entropy_weight', 2.0, 'cross_entropy_weight')
print(' lzx_debug loss_type:', FLAGS.loss_type)
print(' lzx_debug shuffle_buffer_size:', FLAGS.shuffle_buffer_size)
print(' lzx_debug cross_entropy_weight:', FLAGS.cross_entropy_weight)

# --------------- init path ------------------------
# 处理离线训练数据

if FLAGS.is_offline:
    FLAGS.train_paths = FLAGS.train_paths
    FLAGS.test_paths = FLAGS.test_paths

print(' lzx_debug [test] train_paths: ', FLAGS.test_paths)
print(' lzx_debug [test] test_path: ', FLAGS.test_paths)

deep_insight_v2.reset(deep_insight_sample_ratio=FLAGS.deep_instance_sample_ratio)

SHARED_DNN_DIMS_COMMON = eval(FLAGS.shared_dnn_hidden_dims_common)
DNN_DIMS_COMMON = eval(FLAGS.dnn_hidden_dims_common) + [1]
SMALL_DNN_DIMS_COMMON = eval(FLAGS.small_dnn_hidden_dims_common) + [1]
BIAS_DIMS_COMMON = eval(FLAGS.bias_hidden_dims_common) + [1]

EME_DIM = FLAGS.embedding_dim # dense特征的embedding长度

TASK_NAME = ["convert","roi1"]
TASK_NAMES_DICT = {"convert": 0, "roi1": 2}
# TASK_NAMES_DICT = {"convert": 4, "cost": 5}
# TASK_NAME = ["convert","cost","convert_per_hour_label","cost_per_hour_label","convert_per_ad_hour_label","cost_per_ad_hour_label"]
# TASK_NAMES_DICT = {"convert": 0,"cost": 1,"convert_per_hour_label": 2,"cost_per_hour_label": 3,"convert_per_ad_hour_label": 4,"cost_per_ad_hour_label": 5}

def get_real_loss(labels, preds):
    """计算标签与预测值之间的均方误差损失"""
    return tf.square(labels-preds)

# features 输入的是tensorflow  tensor_dict
def supervised_model_fn(model, features, labels, mode, params, config):
    """TensorFlow Estimator模型函数
    功能: 定义模型的训练、评估和预测逻辑, 包括特征处理、模型构建、损失计算和指标收集
    输入: 
        model: SparseEstimator模型实例
        features: 输入特征字典
        labels: 标签数据
        mode: 运行模式( 训练、评估、预测) 
        params: 模型参数
        config: 运行配置
    输出: 
        tf.estimator.EstimatorSpec: 包含模型运行所需的所有信息
    """
    logging_hook_di = dict()
    tf.logging.info("features keys:{}".format(features.keys()))
    # sparse.feature.FeatureSlot初始化设置, serving和training用户的特征是一样的
    sparse.feature.FeatureSlot.set_default_bias_initializer(tf.zeros_initializer())
    sparse.feature.FeatureSlot.set_default_vec_initializer(tf.random_uniform_initializer(-0.0078125, 0.0078125))
    sparse.feature.FeatureSlot.set_default_bias_optimizer(
        sparse.optimizer.SparseFtrlOptimizer(learning_rate=FLAGS.learning_rate,
                                             l1_regularization_strength=1.2,
                                             l2_regularization_strength=0.01))
    sparse.feature.FeatureSlot.set_default_vec_optimizer(
        sparse.optimizer.SparseAdagradOptimizer(learning_rate=FLAGS.learning_rate))

    state_embeddings_names = []
    camp_vmid_feat_hourly = list() # 未被使用

    # embeding维度
    sparse_emb_dim = FLAGS.sparse_emb_dim
    sparse_cross_emb_dim = FLAGS.sparse_cross_emb_dim
    
    # 字典收集不同任务下的embedding列表
    state_embeddings_mt = defaultdict(list)
    lhuc_input_tensors_mt = defaultdict(list)
    cross_embeddings_mt = defaultdict(list)
    campaign_vmid_embs_mt = defaultdict(list)
    campaign_dense_embs_mt = defaultdict(list)

    # 特征长度
    lhuc_first_dim = 0
    cross_first_dim = 0
    campaign_vmid_emb_size = 0
    campaign_dense_embs_size = 0

    # 特征字典, key为特征名称, value为分桶边界值
    sparse_features = SPARSE_FEAT_v2
    dense_features_1d = DENSE_FEAT_1D_v2
    dense_features_2d = DENSE_FEAT_2D_more

    # key为需要log化的特征名, value为log化后的分桶边界值
    need_log_feature = dict()
    if FLAGS.use_log_feature:
        need_log_feature = NEED_LOG_FEATURE_DICT

    # 三元组( 'name_1', 'name_2', 'fc_name') , 'fc_name'的特征 = 'name_1'特征 / 'name_2'特征
    ratio_list = RATIO_FEATURE
    # key为上述组合出的需要log化的'fc_name'特征名, value为log化后的分桶边界值
    ratio_feat_info = RATIO_FEATURE_INFO

    # key为需要delta操作的特征前缀名, value是特征的时长范围列表, 有效的特征名={特征前缀名}_{时长范围}h_{特征类型}_all
    delta_camp = DELTA_CAMP_FEAT
    delta_camp_vmid = DELTA_CAMP_VMID_FEAT
    delta_vmid = DELTA_VMID_FEAT
    # key对上述有效的特征名在不同时间范围的均值下进行delta操作的delta特征, value为delta特征log化后的分桶边界值
    delta_feat_info = DELTA_FEATURE_INFO

    # DENSE_FEAT_2D_more二维特征中能找到camp和ad的特征映射关系的特征组合, 后续对特征值做camp=camp-relu(ad)的操作, 疑问: 为什么只对2d做？为什么是这个操作？
    # camp_to_ad = dict((k, CAMP_FEAT_TO_AD_FEAT_more[k]) for k in dense_features_2d.keys()
    #                if k in CAMP_FEAT_TO_AD_FEAT_more)

    # assert EXTRA_FEAT[0] == "vmid_ea_dea_dbt_aid_cnt"

    # adid_cnt_feature_ = features[EXTRA_FEAT[0]] #tf.feature_column.input_layer(features, utils.DenseFeature(name="groupvalidadcnt", dtype=tf.int64, default_value=0).get_feature_column())
    # adid_cnt_feature = bucket_single_feat_semantic(adid_cnt_feature_,bucket_points=ADCNT_BUCKET_POINTS,dim=FLAGS.adcnt_embedding_dim,method='dis',fc='fc_dense_delivery_ad_num', suffix="common",temp=FLAGS.temp)
    # adcnt_add_1_ =  tf.add(adid_cnt_feature_, tf.ones_like(adid_cnt_feature_))
    # adcnt_add_1 = bucket_single_feat_semantic(adcnt_add_1_,bucket_points=ADCNT_BUCKET_POINTS,dim=FLAGS.adcnt_embedding_dim,method='dis',fc='fc_dense_delivery_ad_num_add_1', suffix="common",temp=FLAGS.temp)
    # adcnt_add_2_ =  tf.add(adid_cnt_feature_, 2*tf.ones_like(adid_cnt_feature_))
    # adcnt_add_2 = bucket_single_feat_semantic(adcnt_add_2_,bucket_points=ADCNT_BUCKET_POINTS,dim=FLAGS.adcnt_embedding_dim,method='dis',fc='fc_dense_delivery_ad_num_add_2', suffix="common",temp=FLAGS.temp)
    # adcnt_add_4_ =  tf.add(adid_cnt_feature_, 4*tf.ones_like(adid_cnt_feature_))
    # adcnt_add_4 = bucket_single_feat_semantic(adcnt_add_4_,bucket_points=ADCNT_BUCKET_POINTS,dim=FLAGS.adcnt_embedding_dim,method='dis',fc='fc_dense_delivery_ad_num_add_4', suffix="common",temp=FLAGS.temp)
    # adcnt_add_8_ =  tf.add(adid_cnt_feature_, 8*tf.ones_like(adid_cnt_feature_))
    # adcnt_add_8 = bucket_single_feat_semantic(adcnt_add_8_,bucket_points=ADCNT_BUCKET_POINTS,dim=FLAGS.adcnt_embedding_dim,method='dis',fc='fc_dense_delivery_ad_num_add_8', suffix="common",temp=FLAGS.temp)
    
    # 稀疏特征处理: 将稀疏特征转换为embedding表示
    for feat_name, slot_id_hash_size in sorted(sparse_features.items()):
        slot_id, hash_size = slot_id_hash_size
        # 创建特征槽位, 用于存储特征的嵌入权重
        fs = model.add_feature_slot(slot_id, hash_size)
        # 根据槽位ID选择不同的特征列添加方式
        if slot_id < 1024:
            # 旧版本特征列添加方式
            fc = model.add_feature_column_v1(fs)
        else:  # fid v2 版本
            # 新版本特征列添加方式, 直接使用特征名称
            fc = model.add_feature_column_v2(feat_name, fs)
        # 收集特征, 累计维度
        if FLAGS.separate_embedding:
            # 使用分任务的embedding, 每个任务有独立的特征嵌入
            for t in TASK_NAME:
                state_embeddings_mt[t].append(fc.add_vector(sparse_emb_dim))
                lhuc_input_tensors_mt[t].append(fc.add_vector(sparse_emb_dim))
                cross_embeddings_mt[t].append(fc.add_vector(sparse_cross_emb_dim))
                print(" lzx_debug len state_embeddings_mt[%s]: %d" % (t, len(state_embeddings_mt[t])))
            cross_first_dim += sparse_cross_emb_dim
            lhuc_first_dim += sparse_emb_dim
        else:
            # 使用公共embedding, 所有任务共享相同的特征嵌入
            common_emb = fc.add_vector(sparse_emb_dim)
            lhuc_common_emb = fc.add_vector(sparse_emb_dim)
            cross_common_emb = fc.add_vector(sparse_cross_emb_dim)
            for t in TASK_NAME:
                state_embeddings_mt[t].append(common_emb)
                lhuc_input_tensors_mt[t].append(lhuc_common_emb)
                cross_embeddings_mt[t].append(cross_common_emb)
                print(" lzx_debug len state_embeddings_mt[%s]: %d" % (t, len(state_embeddings_mt[t])))
            cross_first_dim += sparse_cross_emb_dim
            lhuc_first_dim += sparse_emb_dim
        # 收集特征名
        state_embeddings_names.append(feat_name)
    
    # 对camp_to_ad特征值做camp=camp-relu(ad)的操作
    # for k, v in camp_to_ad.items():
    #     features[k] = features[k] - tf.nn.relu(features[v])
    print(" lzx_debug state_embeddings_names:", state_embeddings_names)
    print(" lzx_debug len sparse_features:", len(state_embeddings_names))
    # 稠密特征处理: 包括特征分桶和embedding转换
    with tf.variable_scope("model_cur", reuse=tf.AUTO_REUSE,
                           partitioner=tf.fixed_size_partitioner(FLAGS.ps_num_embedding_shards, axis=0)):
        # 对需要log化的特征做log变换, 对2D特征做最大值池化, 对所有特征做分桶和embedding
        dense_features_1d_embeddings, dense_features_1d_embedding_names, _, _, _ = bucket_feats(
            features,  # 输入特征字典
            dense_features_1d,  # 一维稠密特征配置
            log_dict=need_log_feature,  # 需要log变换的特征配置
            need_log1p=False,  # 是否使用log(1+x)变换
            dim=EME_DIM,  # embedding维度
            all_feat_suffix="common"  # 特征名后缀
        )
        # 收集特征名
        state_embeddings_names += dense_features_1d_embedding_names
        # 分任务收集特征, 累计维度
        for t in TASK_NAME:
            state_embeddings_mt[t] += dense_features_1d_embeddings
            campaign_dense_embs_mt[t] += dense_features_1d_embeddings
            cross_embeddings_mt[t] +=  dense_features_1d_embeddings
            print(" lzx_debug len state_embeddings_mt[%s]: %d" % (t, len(state_embeddings_mt[t])))
        lhuc_first_dim += EME_DIM * len(dense_features_1d_embedding_names) #疑问: 这里没收集lhuc的特征为什么加维度？
        campaign_dense_embs_size += EME_DIM * len(dense_features_1d_embedding_names)
        cross_first_dim += EME_DIM * len(dense_features_1d_embedding_names)
        print(" lzx_debug len dense_features_1d_names:", len(dense_features_1d_embedding_names))

        # 处理二维稠密特征, 包括比率特征计算、差值特征计算、序列池化等
        dense_features_2d_embeddings, dense_features_2d_names, all_emb_size, \
        campaign_embeddings, campaign_vmid_embedding, vmid_emb_size = bucket_feats_2d(
            features,  # 输入特征字典
            dense_features_2d,  # 二维稠密特征配置
            log_dict=need_log_feature,  # 需要log变换的特征配置
            need_log1p=False,  # 是否使用log(1+x)变换
            dim=EME_DIM,  # embedding维度
            need_reduce="max",  # 池化方式：最大值池化
            all_feat_suffix="common",  # 特征名后缀
            seq_pooling=FLAGS.seq_pooling,  # 是否使用序列池化
            ratio_feat_list=ratio_list,  # 比率特征列表
            ratio_feat=ratio_feat_info,  # 比率特征配置
            delta_camp=delta_camp,  # 差值特征(广告组级别)
            delta_camp_vmid=delta_camp_vmid,  # 差值特征(广告组-媒体级别)
            delta_vmid=delta_vmid,  # 差值特征(媒体级别)
            delta_feat=delta_feat_info,  # 差值特征配置
            din=FLAGS.use_din  # 是否使用DIN(Deep Interest Network)
        )
        tmp_camp_emb_size = EME_DIM * len(campaign_embeddings)

        for t in TASK_NAME:
            state_embeddings_mt[t] += dense_features_2d_embeddings
            campaign_dense_embs_mt[t] += campaign_embeddings
            campaign_vmid_embs_mt[t].append(campaign_vmid_embedding)
            print(" lzx_debug len state_embeddings_mt[%s]: %d" % (t, len(state_embeddings_mt[t])))

        campaign_vmid_emb_size += vmid_emb_size
        campaign_dense_embs_size += tmp_camp_emb_size
        state_embeddings_names += dense_features_2d_names
        lhuc_first_dim += all_emb_size
        print(" lzx_debug len dense_features_2d_names:", len(dense_features_2d_names))
        print(" lzx_debug len state_embeddings_names(eqal_sparse+dense1d+dense2d):", len(state_embeddings_names))

    # 打印tensorboard
    for t in TASK_NAME:
        assert len(state_embeddings_mt[t]) == len(
            state_embeddings_names), "len_state_embeddings_mt_{}: {} \t len_state_embeddings_names: {}".format(
            t, len(state_embeddings_mt[t]), len(state_embeddings_names)
        )
        is_all_fea_same = dict()
        for idx in range(len(state_embeddings_names)):
             # 记录各个特征, 每个batch内是否相同
            feat_mean = tf.reduce_sum(state_embeddings_mt[t][idx], axis = 1)
            tf.summary.histogram(
                'sum_{}_{}'.format(t, state_embeddings_names[idx]), feat_mean
            )
            tf.summary.histogram(
                'feat_{}_{}'.format(t, state_embeddings_names[idx]), state_embeddings_mt[t][idx]
            )
        print(' lzx_debug feat_mean.shape:', feat_mean.shape)

    # 跨任务特征处理: 整合不同任务的共享特征
    state_embedding_mt = dict()  # 主特征嵌入字典, 存储每个任务的主要特征向量
    vmid_dense_embedding_mt = dict()  # 媒体ID相关稠密特征嵌入字典
    campaign_dense_embedding_mt = dict()  # 广告组相关稠密特征嵌入字典
    lhuc_embedding_mt = dict()  # LHUC控制特征嵌入字典, 用于动态调整特征权重
    cross_embedding_mt = dict()  # 交叉特征嵌入字典, 用于存储特征交叉后的表示
    cross_embedding_merge_vmid_mt = dict()  # 合并媒体ID的交叉特征嵌入字典

    pred_mt = dict()
    # 对每个任务的特征进行拼接, 合并成完整的特征向量
    for t in TASK_NAME:
        state_embedding_mt[t] = tf.concat(state_embeddings_mt[t], axis=1)
        print(' lzx_debug state_embedding_mt[t].shape', state_embedding_mt[t].shape)
        pred_mt['feature_num'] = len(state_embeddings_mt[t])
        # vmid_dense_embedding_mt[t] = campaign_vmid_embs_mt[t][-1]
        # merge_ = cross_embeddings_mt[t] + campaign_vmid_embs_mt[t]
        # cross_embedding_merge_vmid_mt[t] = tf.concat(merge_, axis=1)
        # campaign_dense_embedding_mt[t] = tf.concat(campaign_dense_embs_mt[t], axis=1)
        # cross_embedding_mt[t] = tf.concat(cross_embeddings_mt[t], axis=1)
        # print(" lzx_debug task {}  cross embedding dimensionality: {} : first_dim: {}".format(
        #     t, cross_embedding_mt[t].get_shape(), cross_first_dim))
        # print(" lzx_debug task {}  State embedding dimensionality: {}: lhuc_first_dim first dim: {}\n "
        #       "small lhuc: {}\t small lhuc emb_size: {}\t campaign_dense_embedding_mt: {}".format(
        #     t, state_embedding_mt[t].get_shape(), lhuc_first_dim,
        #     vmid_dense_embedding_mt[t].get_shape(), campaign_vmid_emb_size,
        #     campaign_dense_embedding_mt[t].get_shape()))
        lhuc_embedding_mt[t] = tf.concat(lhuc_input_tensors_mt[t], axis=1)
        # print(" lzx_debug task {}  lhuc embedding dimensionality: {}".format(t, lhuc_embedding_mt[t].get_shape()))

    with tf.variable_scope("model_cur", reuse=tf.AUTO_REUSE,
                           partitioner=tf.fixed_size_partitioner(FLAGS.ps_num_embedding_shards, axis=0)
                           ):
            # [batch, 1]
            logit_mt = list()
            logit_mt_dict = dict()
            is_train = (mode == tf.estimator.ModeKeys.TRAIN)
            # print(" lzx_debug lhuc_first_dim: {} \t use_lhuc: {}", lhuc_first_dim, FLAGS.use_lhuc)
            for task in TASK_NAME:
                state_tensor = state_embedding_mt[task]
                lhuc_tensor = lhuc_embedding_mt[task]
                name = "all"

                # 计算模型输出logit值: 根据特征计算最终预测结果
                logit_list = list()
                if FLAGS.use_lhuc:
                    print(" lzx_debug lzx_debug: use lhuc")
                    # 使用LHUC(Learnable Hierarchical Unit Controller)塔结构, 支持动态调整特征权重
                    state_embedding_logit = get_lhuc_out(
                        state_tensor,  # 主特征张量
                        DNN_DIMS_COMMON, "task_{}_score_{}".format(task, name),
                        lhuc_tensor,  # LHUC控制张量
                        first_dim=int(state_tensor.shape[1]),
                        concat_nn_input=False,
                        enable_bias=True,
                        lhuc_bottle_neck_dim=32,  # LHUC瓶颈层维度
                        is_train=is_train,
                        dropout_prob=FLAGS.dropout_prob,
                        task = task
                    )
                else:
                    # senet, 得到所有特征的重要度
                    print("state_tensor 形状:", state_tensor.shape)  # 查看原始形状
                    print("emb_num = len(state_embeddings_mt[task]):", len(state_embeddings_mt[task]))  # 确认 emb_num 是否为 897
                    print(" lzx_debug lzx_debug: not use lhuc")
                    # 使用标准密集神经网络塔结构
                    state_embedding_logit = get_dense_tower(
                        DNN_DIMS_COMMON,
                        state_tensor,
                        "task_{}_score_{}".format(task, name),
                        is_train=is_train,
                        dropout_prob=FLAGS.dropout_prob
                    )
                state_embedding_logit = tf.nn.softmax(state_embedding_logit)
                state_embedding_logit = tf.reshape(state_embedding_logit, [-1])
                print("state_embedding_logit.shape: ", state_embedding_logit.shape)
                logit_list.append(state_embedding_logit)

                state_embedding_logit_merge = logit_list[0]
                logit_mt.append(state_embedding_logit_merge)
                logit_mt_dict[task] = state_embedding_logit_merge
                
                pred_mt[task] = state_embedding_logit     
    
    model.freeze_slots(features)
    # 构建损失函数: 根据不同任务类型计算相应的损失
            # 添加模型参数
    if mode == tf.estimator.ModeKeys.PREDICT:
        '''
        注释掉的是旧版本的预测模式实现, 包含权重调整和sigmoid激活
        '''
        # 构建预测结果字典
        predictions_dict = {}
        # 添加每个任务的预测结果
        for task in TASK_NAME:
            predictions_dict[task] = pred_mt[task]
        # 特别添加成本可控性预测结果
        predictions_dict['is_cost_controllable'] = pred_mt['is_cost_controllable']
                    
        # 返回预测模式的EstimatorSpec
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict)
    else:
        loss_merge = 0
        eval_tensors = dict()
        score_mt_dict = dict()
        label_mt_dict = dict()
        value_mt_dict = dict()
        logit_post_mt_dict = dict()

        label_mt = {}
        loss_mt = {}

        for task, idx in TASK_NAMES_DICT.items():
            # if idx == 0 and FLAGS.enable_ad_avg_uplift:
            #     cnt_clip = tf.cast(tf.maximum(labels[:, 6], 1), tf.float32)
            #     label_mt[task] = labels[:, idx] / cnt_clip
            # else:
            #     label_mt[task] = labels[:, idx]
            label_mt[task] = labels[:, idx]

        for task in TASK_NAME:
            task_score = tf.reshape(pred_mt[task], [-1])
            # 对 task_score 进行 归一化 操作
            # 计算 task_score 的最小值和最大值
            # task_score = tf.nn.sigmoid(task_score)
            # min_score = tf.reduce_min(task_score)
            # max_score = tf.reduce_max(task_score)
            # 进行线性归一化
            # task_score = (task_score - min_score) / (max_score - min_score)
            task_label = label_mt[task]
            task_label = tf.reshape(tf.clip_by_value(task_label, FLAGS.label_cap_value_lb, FLAGS.label_cap_value_ub), [-1])
            label_masks = task_label>=1
            task_label = tf.cast(label_masks, tf.float32)
            # 根据配置选择不同的损失函数
            if FLAGS.loss_type == 'mae':
                # 平均绝对误差(MAE)损失
                loss_pre = tf.abs(task_label - task_score)
                loss = tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss:mae")
            elif FLAGS.loss_type == 'huber':
                # Huber损失, 结合了MAE和MSE的优点
                loss_pre = huber_loss(task_label, task_score, FLAGS.huber_delta)
                loss = tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss:huber_loss")
            elif FLAGS.loss_type == 'weighted_cross_entropy':
                # 加权交叉熵损失, 适用于类别不平衡问题
                loss_pre = weighted_cross_entropy(task_label, task_score, FLAGS.cross_entropy_weight, logging_hook_di)
                loss = tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss:weighted_cross_entropy")
            elif FLAGS.loss_type == 'cross_entropy':
                # 标准交叉熵损失
                loss_pre = cross_entropy(task_label, task_score, logging_hook_di)
                loss = tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss:cross_entropy")
            else:
                # 默认使用均方误差(MSE)损失
                loss_pre = get_real_loss(task_label, task_score)
                loss = tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss: mse")
            logging_hook_di['[lzx_debug] reduce_mean_loss'] = loss
            loss_mt[task] = loss
            loss_merge += loss
            logging_hook_di['[lzx_debug] loss_merge'] = loss_merge
            with tf.name_scope("Loss"):
                tf.summary.scalar(
                    "loss_mean_auto_{}".format(task), loss
                )

            metrics_dict = dict()
            with tf.name_scope("Metric"):
                # metrics_dict['loss'] = loss
                #metrics_dict['mae_{}'.format(task)] = get_mae(task_score, task_label)
                #metrics_dict['mse_{}'.format(task)] = get_mse(task_score, task_label)
                metrics_dict['label_{}'.format(task)] = tf.reduce_mean(task_label)
                metrics_dict['score_{}'.format(task)] = tf.reduce_mean(task_score)

            eval_tensors.update(metrics_dict)
            

            #loss = tf.reduce_sum(loss_pre)
            #loss_merge += loss

        req_time_tensor = tf.fill(tf.shape(logit_mt[0]), tf.timestamp(name=None))
        mock_user = tf.fill(tf.shape(logit_mt[0]), 0)

        eval_metric_prediction = CustomMetricHook(
            eval_tensors, log_steps=FLAGS.save_summary_steps, ema_decay=FLAGS.ema_decay)
        for k, v in metrics_dict.items():
            tf.summary.scalar(
                k, tf.reduce_mean(v)
            )
        scaffold = tf.compat.v1.train.Scaffold()
        if mode == tf.estimator.ModeKeys.EVAL:
            loss_eval = loss_merge
            eval_summary_hook = tf.estimator.SummarySaverHook(
                save_steps=20,
                output_dir=os.path.join(JOB_CONTEXT.summary_dir, 'eval'),
                scaffold=scaffold
            )
            save_scores_tensor_dict = dict()
            save_labels_tensor_dict = dict()
            save_task_names = TASK_NAME

            # 准备保存的标签和预测分数张量字典
            for task_name in save_task_names:
                # 原始标签统计
                save_labels_tensor_dict[task_name] = tf.reshape(label_mt[task_name], [-1])
                # 原始预测分数统计
                save_scores_tensor_dict[task_name] = tf.reshape(pred_mt[task_name], [-1])

                # 对标签进行截断(cap)处理, 限制在[0, 300]范围内, 防止极端值影响模型训练
                # 截断后的标签统计
                save_labels_tensor_dict[task_name+"_cap"] = tf.reshape(tf.clip_by_value(label_mt[task_name], 0, 300), [-1])
                # 预测分数统计(与原始相同)
                save_scores_tensor_dict[task_name+"_cap"] = tf.reshape(pred_mt[task_name], [-1])


                '''
                score = score_mt_dict[task_name]
                logit_post = logit_post_mt_dict[task_name]
                is_auto_type = tf.reshape(is_auto_type, [-1])
                save_scores_tensor_dict['class_{}'.format(task_name)] = tf.reshape(score, [-1])
                save_scores_tensor_dict['regression_{}'.format(task_name)] = tf.reshape(logit_post, [-1])
                save_scores_tensor_dict['is_eval_{}'.format(task_name)] = tf.reshape(
                    tf.ones_like(logit_post, tf.float32),
                    [-1]
                )

                save_labels_tensor_dict['class_{}'.format(task_name)] = tf.reshape(label_mt_dict[task_name], [-1])
                save_labels_tensor_dict['regression_{}'.format(task_name)] = tf.reshape(value_mt_dict[task_name], [-1])
                save_labels_tensor_dict['is_eval_{}'.format(task_name)] = tf.reshape(
                    tf.ones_like(logit_post, tf.float32),
                    [-1]
                )
                '''
            
            # 创建DeepInsight2钩子, 用于日志记录和监控评估指标
            di2_multihead_hook = train.DeepInsight2MultiHeadHook(
                tf.reshape(mock_user, [-1]),  # 用户ID( 这里使用占位符) 
                tf.reshape(req_time_tensor, [-1]),  # 请求时间
                score_tensor_dict=save_scores_tensor_dict,  # 预测分数张量字典
                label_tensor_dict=save_labels_tensor_dict,  # 真实标签张量字典
                extra_tensors={  # 额外需要记录的张量信息
                    'dataset': tf.fill(tf.shape(logit_mt[0]), 'eval'),  # 数据集类型
                    'req_id': tf.reshape(features['req_id'], [-1]),  # 请求ID
                    'customer_id': tf.reshape(features['customer_id'], [-1]),  # 客户ID
                    'advertiser_id': tf.reshape(features['advertiser_id'], [-1]),  # 广告商ID
                    # 'campaign_id': tf.reshape(features['campaign_id'], [-1]),
                    'external_action': tf.reshape(features['external_action'], [-1]),  # 外部行为类型
                    'deep_external_action': tf.reshape(features['deep_external_action'], [-1]),  # 深度外部行为
                    'deep_bid_type': tf.reshape(features['deep_bid_type'], [-1]),  # 竞价类型
                    'app_package': tf.reshape(features['app_package'], [-1]),  # 应用包名
                    'target_app_package': tf.reshape(features['target_app_package'], [-1]),  # 目标应用包名
                    'fc_real_cost_72h_vmid_all': tf.reshape(features['fc_real_cost_72h_vmid_all'], [-1]),  # 72小时真实成本特征
                    'fc_pvr_72h_vmid_all': tf.reshape(features['fc_pvr_72h_vmid_all'], [-1]),  # 72小时PVR特征
                    'cost_label': tf.reshape(labels[:, 0], [-1]),  # 成本标签
                    'convert_label': tf.reshape(labels[:, 1], [-1]),  # 转化标签
                    'advv_label': tf.reshape(labels[:, 2], [-1]),  # 广告价值标签
                    'sum_pvr_label': tf.reshape(labels[:, 3], [-1]),  # 总PVR标签
                    'cost_controllable_label': tf.reshape(labels[:, 4], [-1]),  # 成本可控标签
                    'nobid_cost_label': tf.reshape(labels[:, 5], [-1]),  # 无竞价成本标签
                    'p_date': tf.reshape(labels[:, 6], [-1]),  # 日期特征
                },
                neg_sample_rate=1.0  # 负样本采样率, 1.0表示不进行采样
            )
            update_metric_tensors = dict()
            for k, v in eval_tensors.items():
                update_metric_tensors[k] = tf.metrics.mean(v)

            for task_name in TASK_NAME:
                # 计算准确率指标
                update_metric_tensors["acc_{}".format(task_name)] = tf.metrics.accuracy(
                    labels=save_labels_tensor_dict[task_name],  # 真实标签
                    predictions=save_scores_tensor_dict[task_name],  # 预测值
                    name='acc_op_{}'.format(task_name))  # 指标操作名称

                # 计算ROC曲线下面积(AUC)指标
                update_metric_tensors["auc_{}".format(task_name)] = tf.metrics.auc(
                    labels=save_labels_tensor_dict[task_name],  # 真实标签
                    predictions=save_scores_tensor_dict[task_name],  # 预测分数
                    weights=None,  # 不使用样本权重
                    num_thresholds=200,  # 阈值数量, 用于近似计算AUC
                    metrics_collections=None,
                    updates_collections=None,
                    curve='ROC',  # 计算ROC曲线下面积
                    summation_method='trapezoidal',  # 积分方法：梯形法则
                        thresholds=None,  # 自动选择阈值
                        name='auc_op_{}'.format(task_name))  # 指标操作名称

                    # 使用AUC函数计算特定阈值下的召回率
                update_metric_tensors["recall_{}".format(task_name)] = tf.metrics.auc(
                    labels=save_labels_tensor_dict[task_name],  # 真实标签
                    predictions=save_scores_tensor_dict[task_name],  # 模型预测值
                    thresholds=[0.08, 0.1, 0.2, 0.5],  # 指定多个阈值点计算召回率
                    weights=None,
                    metrics_collections=None, updates_collections=None,
                    name='recall_op_' + task_name
                )

                update_metric_tensors["pred_mean_{}".format(task_name)] = tf.metrics.mean(
                    values=save_scores_tensor_dict[task_name],
                    weights=None, metrics_collections=None,
                    updates_collections=None, name='predict_mean_op_' + task_name)

                update_metric_tensors["label_mean_{}".format(task_name)] = tf.metrics.mean(
                    values=save_labels_tensor_dict[task_name],
                    weights=None, metrics_collections=None,
                    updates_collections=None, name='label_mean_op_' + task_name)

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss_eval,
                eval_metric_ops=update_metric_tensors,
                evaluation_hooks=[eval_metric_prediction, eval_summary_hook, di2_multihead_hook])
        else:
            save_scores_tensor_dict = dict()
            save_labels_tensor_dict = dict()
            save_task_names = TASK_NAME
            for task_name in save_task_names:
                # label统计
                save_labels_tensor_dict[task_name] = tf.reshape(label_mt[task_name], [-1])
                # predict统计
                save_scores_tensor_dict[task_name] = tf.reshape(pred_mt[task_name], [-1])

                # cap
                # label统计
                save_labels_tensor_dict[task_name+"_cap"] = tf.reshape(tf.clip_by_value(label_mt[task_name], 0, 300), [-1])
                # predict统计
                save_scores_tensor_dict[task_name+"_cap"] = tf.reshape(pred_mt[task_name], [-1])

                '''
                注释掉的代码段是旧版本的指标统计实现, 包含分类和回归两种预测结果的记录
                '''

            # 创建训练模式下的DeepInsight2钩子, 用于日志记录和监控训练过程
            di2_multihead_hook = train.DeepInsight2MultiHeadHook(
                tf.reshape(mock_user, [-1]),  # 用户ID( 这里使用占位符) 
                tf.reshape(req_time_tensor, [-1]),  # 请求时间
                score_tensor_dict=save_scores_tensor_dict,  # 预测分数张量字典
                label_tensor_dict=save_labels_tensor_dict,  # 真实标签张量字典
                extra_tensors={  # 额外需要记录的张量信息
                    'dataset': tf.fill(tf.shape(logit_mt[0]), 'train'),  # 数据集类型标记为训练集
                    'req_id': tf.reshape(features['req_id'], [-1]),  # 请求ID
                    'customer_id': tf.reshape(features['customer_id'], [-1]),  # 客户ID
                    'advertiser_id': tf.reshape(features['advertiser_id'], [-1]),  # 广告商ID
                    # 'campaign_id': tf.reshape(features['campaign_id'], [-1]),
                    'external_action': tf.reshape(features['external_action'], [-1]),  # 外部行为类型
                    'deep_external_action': tf.reshape(features['deep_external_action'], [-1]),  # 深度外部行为
                    'deep_bid_type': tf.reshape(features['deep_bid_type'], [-1]),  # 竞价类型
                    'app_package': tf.reshape(features['app_package'], [-1]),  # 应用包名
                    'target_app_package': tf.reshape(features['target_app_package'], [-1]),  # 目标应用包名
                    'fc_real_cost_72h_vmid_all': tf.reshape(features['fc_real_cost_72h_vmid_all'], [-1]),  # 72小时真实成本特征
                    'fc_pvr_72h_vmid_all': tf.reshape(features['fc_pvr_72h_vmid_all'], [-1]),  # 72小时PVR特征
                    'cost_label': tf.reshape(labels[:, 0], [-1]),  # 成本标签
                    'convert_label': tf.reshape(labels[:, 1], [-1]),  # 转化标签
                    'advv_label': tf.reshape(labels[:, 2], [-1]),  # 广告价值标签
                    'sum_pvr_label': tf.reshape(labels[:, 3], [-1]),  # 总PVR标签
                    'cost_controllable_label': tf.reshape(labels[:, 4], [-1]),  # 成本可控标签
                    'nobid_cost_label': tf.reshape(labels[:, 5], [-1]),  # 无竞价成本标签
                    'p_date': tf.reshape(labels[:, 6], [-1]),  # 日期特征
                },
                neg_sample_rate=1.0  # 负样本采样率, 1.0表示不进行采样
            )
        # 创建优化器, 使用Adagrad算法, 学习率从FLAGS配置中获取
        main_opt = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate)
        # 获取除model_bias作用域外的所有变量, 用于梯度计算
        main_vars = get_vars_not_in_scope("model_bias")
        # 计算梯度, 使用总的损失函数loss_merge对所有主变量计算梯度
        main_grads_and_vars = main_opt.compute_gradients(loss=loss_merge, var_list=main_vars)
        # 记录梯度直方图, 用于TensorBoard可视化和模型调试
        for v, (grad, val) in zip(main_vars, main_grads_and_vars):
            if grad is not None:
                tf.summary.histogram("grad_{}".format(v), grad)
        # 应用梯度更新参数, 并更新全局步数计数器
        train_op = main_opt.apply_gradients(main_grads_and_vars, global_step=tf.train.get_global_step())
        
        logging_hook = tf.train.LoggingTensorHook(tensors=logging_hook_di, every_n_iter=1)
        synced_saver_hook = SynchronizedCheckpointSaverHook(config, scaffold)
        training_chief_hooks = [tf.train.ProfilerHook(save_steps=1000)]
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss_merge,
                                          train_op=train_op,
                                          scaffold=scaffold,
                                          training_hooks=[synced_saver_hook, di2_multihead_hook, logging_hook],
                                          training_chief_hooks=training_chief_hooks)


def get_estimator():
    """创建TensorFlow Estimator实例
    
    功能: 配置并创建SparseEstimator实例, 支持从检查点恢复模型
    输入: 无( 使用全局FLAGS配置) 
    输出: 
        estimator: SparseEstimator实例
        config: 运行配置对象
    """
    # 检查点目录路径
    checkpoint_dir = os.path.join(FLAGS.model_path, 'checkpoints')
    if FLAGS.last_model_path and not tf.train.latest_checkpoint(checkpoint_dir):
        if FLAGS.batch_reload:
            warmup_dir = os.path.join(FLAGS.last_model_path, 'checkpoints')
            print(" lzx_debug input last_model_path: {} \t do reload!".format(checkpoint_dir))
        else:
            warmup_dir = None
            print(" lzx_debug input last_model_path: {} \t but do not reload!".format(checkpoint_dir))
    else:
        warmup_dir = None
    config = tf.estimator.RunConfig(save_summary_steps=FLAGS.save_summary_steps,
                                    log_step_count_steps=FLAGS.log_step_count_steps,
                                    save_checkpoints_secs=FLAGS.save_checkpoints_secs,
                                    keep_checkpoint_max=FLAGS.keep_checkpoint_max,
                                    model_dir=checkpoint_dir,
                                    tf_random_seed=FLAGS.seed
                                    )
    return sparse.estimator.SparseEstimator(
        model_fn=supervised_model_fn,
        num_embedding_shards=None if FLAGS.is_train == 1 else FLAGS.ps_num_embedding_shards,
        config=config,
        warm_start_from=warmup_dir), config


def train_and_evaluate():
    """训练和评估模型
    
    功能: 创建模型实例, 配置特征, 执行训练和评估过程
    输入: 无( 使用全局FLAGS配置) 
    输出: 
        model: 训练好的模型实例
        run_config: 运行配置对象
    """
    # 获取模型实例和运行配置
    model, run_config = get_estimator()

    # 加载特征配置
    sparse_features = SPARSE_FEAT_v2  # 稀疏特征配置，13个，配置的
    dense_features_1d = DENSE_FEAT_1D_v2  # 一维稠密特征配置，744个
    dense_features_2d = DENSE_FEAT_2D_more  # 二维稠密特征配置，273个


    # 配置一维稠密特征的特征列, 固定长度为1, 默认值为0
    dense_features = {fc: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=[0]) for fc in
                      sorted(dense_features_1d)}

    # 更新二维和时序特征的特征列配置, 固定长度为20, 默认值为20个0
    dense_features.update({fc: tf.io.FixedLenFeature(shape=[20], dtype=tf.int64, default_value=[0] * 20) for fc in
                           sorted(list(dense_features_2d.keys()))})

    # dense_features.update({fc: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32, default_value=[0.0]) for fc in
    #                        sorted(list(NAME_TO_LABEL.values()))})

    num_shards = run_config.num_worker_replicas # 1
    shard_id = run_config.task_id + (0 if run_config.is_chief else 1) # 0

    if FLAGS.is_train == 1:
        print(' lzx_debug Training start!') # batch_size: 256
        model.train(
            input_fn=lambda: create_instance_dataset(FLAGS.train_paths,
                                                     sparse_keys=list(sparse_features.keys()),
                                                     num_shards=num_shards,
                                                     shard_id=shard_id,
                                                     batch_size=FLAGS.batch_size,
                                                     dense_features=dense_features,
                                                     shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                                                     n_epochs=1,
                                                     cycle_length=FLAGS.cycle_length,
                                                     block_length=FLAGS.block_length,
                                                     num_parallel_maps=FLAGS.num_parallel_maps,
                                                     is_auto_type=FLAGS.auto_type_to_keep,
                                                     only_netservice=FLAGS.only_netservice
                                                     ),
        )
        print(' lzx_debug Training finishes!')

    if FLAGS.is_eval == 1:
        print(' lzx_debug Evaluation starts!')
        eval_metrics = model.evaluate(
            input_fn=lambda: create_instance_dataset(FLAGS.test_paths,
                                                     sparse_keys=list(sparse_features.keys()),
                                                     num_shards=num_shards,
                                                     shard_id=shard_id,
                                                     batch_size=FLAGS.batch_size,
                                                     dense_features=dense_features,
                                                     shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                                                     n_epochs=FLAGS.n_epochs,
                                                     cycle_length=FLAGS.cycle_length,
                                                     block_length=FLAGS.block_length,
                                                     num_parallel_maps=FLAGS.num_parallel_maps,
                                                     is_auto_type=FLAGS.auto_type_to_keep,
                                                     only_netservice=FLAGS.only_netservice
                                                     )
            , steps=None
        )
        print(' lzx_debug eval metric_ops = {}'.format(eval_metrics))
        for metric_name, metric_value in eval_metrics.items():
            print(f" lzx_debug {metric_name}: {metric_value}")
        print(' lzx_debug Evaluation finishes!')

    return model, run_config


def serving_input_receiver_fn():
    """创建服务输入接收器
    
    功能: 定义模型服务时的输入特征结构, 用于模型导出
    输入: 无
    输出: 
        tf.estimator.export.ServingInputReceiver: 包含输入特征和接收器的对象
    """
    # 定义稠密特征配置
    dense_features_1d = DENSE_FEAT_1D_v2
    dense_features_2d = DENSE_FEAT_2D_more

    feat_2d = dict((k, CAMP_FEAT_TO_AD_FEAT_more[k]) for k in dense_features_2d.keys() if k in CAMP_FEAT_TO_AD_FEAT_more)

    # 创建服务时的特征占位符
    features_next = {
        # 稀疏特征的索引、值和形状信息, 用于表示SparseTensor
        'fids_indices': tf.placeholder(tf.int64, shape=[None], name='fids_indices'),  # 稀疏特征索引
        'fids_values': tf.placeholder(tf.int64, shape=[None], name='fids_values'),  # 稀疏特征值
        'fids_dense_shape': tf.placeholder(tf.int64, shape=[None], name='fids_dense_shape')  # 稀疏特征的密集形状
    }
    # 区分一维稠密特征中的二维和真正一维特征
    dense_features_1d_real_2d = [k for k in dense_features_1d.keys() if k in CAMP_FEAT_HOURLY]
    dense_features_1d_real_1d = [k for k in dense_features_1d.keys() if k not in CAMP_FEAT_HOURLY]
    
    # 添加二维特征和时序特征的占位符, 形状为[batch_size, 20]
    features_next.update({fc: tf.placeholder(dtype=tf.int64, shape=[None, 20], name=fc) for fc in \
                          sorted(list(dense_features_2d.keys()) + dense_features_1d_real_2d)})
    
    # 添加一维特征和额外特征的占位符, 形状为[batch_size, 1]
    features_next.update({fc: tf.placeholder(dtype=tf.int64, shape=[None, 1], name=fc) for fc in \
                          sorted(dense_features_1d_real_1d)})
    print(" lzx_debug lzx_debug : features_next", features_next)  # 调试信息：打印特征占位符配置
    # 返回服务输入接收器, 同时作为特征和接收特征
    return tf.estimator.export.ServingInputReceiver(features_next, features_next)


def run(_):
    # 执行训练和评估
    model, _ = train_and_evaluate()
    # 只有主节点在训练模式下才导出模型
    if model.config.is_chief and FLAGS.is_train == 1:
        model.export_saved_model(
            export_dir_base=os.path.join(FLAGS.model_path, 'exported'),  # 模型导出路径
            serving_input_receiver_fn=serving_input_receiver_fn  # 服务输入接收器函数
        )


if __name__ == '__main__':
    tf.app.run(run)
