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

from uba_feature_all import DENSE_FEAT_2D_more, SPARSE_FEAT_v2, \
    DENSE_FEAT_1D_v2, EXTRA_FEAT, CAMP_FEAT_TO_AD_FEAT_more, NEED_LOG_FEATURE_DICT

from uba_feature_all_v2 import (RATIO_FEATURE, RATIO_FEATURE_INFO, DELTA_CAMP_FEAT,
                            DELTA_CAMP_VMID_FEAT, DELTA_VMID_FEAT, DELTA_FEATURE_INFO, CAMP_FEAT_HOURLY,
                            CAMP_VMID_FEAT_HOURLY)

from utils.feat_utils import bucket_feats, get_dense_tower, bucket_feats_2d, get_lhuc_tower, get_lhuc_out, print_flags, bucket_single_feat_semantic

from utils.metric_utils import get_AUC
from utils.loss_utils import get_vars_not_in_scope, get_pow_w, huber_loss, weighted_cross_entropy, cross_entropy

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
# tf.app.flags.DEFINE_string('train_paths', 'hdfs://haruna/home/byte_ad_platform/user/liuzixi.123/ecp_roi2_nobid/pid_mid_anchor_model/multi_label_v1/20250314/part-00001.pb.snappy', 'Train path')
# tf.app.flags.DEFINE_string('test_paths', 'hdfs://haruna/home/byte_ad_platform/user/liuzixi.123/ecp_roi2_nobid/pid_mid_anchor_model/multi_label_v1/20250314/part-00001.pb.snappy', 'Test path')
tf.app.flags.DEFINE_string('last_model_path', '', 'last model path.')
tf.app.flags.DEFINE_integer('is_train', 1, 'train mode')
tf.app.flags.DEFINE_integer('is_eval', 0, 'eval mode')
tf.app.flags.DEFINE_integer('ps_num_embedding_shards', 1, 'ps number')
tf.app.flags.DEFINE_integer('batch_size', 256, 'Testing batch size.')
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

# 模型参数
tf.app.flags.DEFINE_integer('seed', 9431, 'random seed')
tf.app.flags.DEFINE_string('dnn_hidden_dims_common', '[512,128,64]', 'DNN hidden dimensionality list in common.')
tf.app.flags.DEFINE_string('small_dnn_hidden_dims_common', '[128,64]', 'small DNN hidden dimensionality list in common.')
tf.app.flags.DEFINE_string('shared_dnn_hidden_dims_common', '[128,64]', 'SHARED DNN hidden dimensionality list in common.')
tf.app.flags.DEFINE_string('bias_hidden_dims_common', '[16, 4]', 'bias hidden dimensionality list 2 in common.')
tf.app.flags.DEFINE_string('need_task', 'convert,cost,convert_per_hour_label,cost_per_hour_label,convert_per_ad_hour_label,cost_per_ad_hour_label', '是否选择某个任务')
tf.app.flags.DEFINE_integer('sparse_emb_dim', '4', '稀疏特征的emb_size')
tf.app.flags.DEFINE_integer('sparse_cross_emb_dim', '2', '稀疏特征的emb_size')
tf.app.flags.DEFINE_integer('embedding_dim', '1', '参数embedding后的维度')
tf.app.flags.DEFINE_float('learning_rate', 0.007, 'Learning rate.')
tf.app.flags.DEFINE_float('ema_decay', 0.99, 'EMA decay for hooks.')
tf.app.flags.DEFINE_float('dropout_prob', 0.0, '是否使用dropout')
tf.app.flags.DEFINE_float('cost_thresh', 100.0, 'cost_level的切换阈值.')
tf.app.flags.DEFINE_bool('separate_embedding', True, '是否使用隔离embedding')
tf.app.flags.DEFINE_bool('separate_dense_embedding', False, '是否使用隔离dense embedding')
tf.app.flags.DEFINE_bool('seq_pooling', True, '是否使用序列感知建模')
tf.app.flags.DEFINE_bool('use_lhuc', False, '是否使用lhuc')
tf.app.flags.DEFINE_bool('use_recify', True, '是否把预估分数纠偏回来')
tf.app.flags.DEFINE_bool('transfer_learning', True, '是否迁移')
tf.app.flags.DEFINE_bool('use_log_feature', True, 'use log feature.')
tf.app.flags.DEFINE_bool('use_din', True, 'use_din')
tf.app.flags.DEFINE_bool('only_netservice', False, '是否只用网服样本')
tf.app.flags.DEFINE_integer('auto_type_to_keep', -1, '是否过滤')
tf.app.flags.DEFINE_bool('enable_ad_avg_uplift', True, 'enable_ad_avg_uplift')
tf.app.flags.DEFINE_float('label_cap_value_lb', -300.0, 'label_cap_value_lb')
tf.app.flags.DEFINE_float('label_cap_value_ub', 300.0, 'label_cap_value_ub') #label的上下界, 当前UBA在脚本里写死了300
tf.app.flags.DEFINE_string('loss_type', 'weighted_cross_entropy', 'loss_type')
tf.app.flags.DEFINE_float("temp", 10, "")
tf.app.flags.DEFINE_integer('adcnt_embedding_dim', 4, 'ad_cnt embedding后的维度')
tf.app.flags.DEFINE_integer('is_offline', 0, 'eval mode')
tf.app.flags.DEFINE_float('cross_entropy_weight', 1.0, 'cross_entropy_weight')
print_flags(FLAGS)
print('is_offline:', FLAGS.is_offline)
print('loss_type:', FLAGS.loss_type)
print('shuffle_buffer_size:', FLAGS.shuffle_buffer_size)
print('cross_entropy_weight:', FLAGS.cross_entropy_weight)

# --------------- init path ------------------------
# 处理离线训练数据
def get_offline_train_eval_path(train_paths):
    train_path_list = train_paths.split('|')
    new_train_path, new_eval_path = [], []
    for train_path in train_path_list:
        base_path_length = len(train_path) - len(train_path.split('/')[-1]) - len(train_path.split('/')[-2]) - 1
        base_path = train_path[:base_path_length]
        rest = train_path[base_path_length:]
        # print('base_path: ', base_path)
        # print('rest: ', rest)
        if rest[-2:] == '/*':
            dates = rest.rstrip('/*').lstrip('{').rstrip('}').split(',')
        else:
            dates = rest.rstrip('/part-*').lstrip('{').rstrip('}').split(',')
            # print('[test2] dates: ', dates)
            # dates = [datetime.strptime(date, '%Y%m%d') for date in dates]
        eval_date = dates[-1]
        if len(dates) == 1:
            train_dates = [(datetime.strptime(eval_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')]
        else:
            train_dates = dates[:-1]
        if rest[-2:] == '/*':
            train_path, eval_path = base_path + '{' + ','.join(train_dates) + '}/*', base_path + '{' + eval_date + '}/*'
        else:
            train_path, eval_path = base_path + '{' + ','.join(train_dates) + '}/part-*', base_path + '{' + eval_date + '}' + '/part-*'
        new_train_path.append(train_path)
        new_eval_path.append(eval_path)
    new_train_path = '|'.join(new_train_path)
    new_eval_path = '|'.join(new_eval_path)
    return new_train_path, new_eval_path

if FLAGS.is_offline:
    # FLAGS.train_paths, FLAGS.test_paths = get_offline_train_eval_path(FLAGS.train_paths)
    FLAGS.train_paths = FLAGS.train_paths
    FLAGS.test_paths = FLAGS.test_paths
    # FLAGS.train_paths = 'hdfs://haruna/home/byte_ad_platform/user/liuzixi.123/ecp_roi2_nobid/pid_mid_anchor_model/multi_label_v1/20250314/part-00001.pb.snappy'
    # FLAGS.test_paths = 'hdfs://haruna/home/byte_ad_platform/user/liuzixi.123/ecp_roi2_nobid/pid_mid_anchor_model/multi_label_v1/20250314/part-00001.pb.snappy'

print('[test] train_paths: ', FLAGS.test_paths)
print('[test] test_path: ', FLAGS.test_paths)
print('[test] is_offline: ', FLAGS.is_offline)

deep_insight_v2.reset(deep_insight_sample_ratio=FLAGS.deep_instance_sample_ratio)

SHARED_DNN_DIMS_COMMON = eval(FLAGS.shared_dnn_hidden_dims_common)
DNN_DIMS_COMMON = eval(FLAGS.dnn_hidden_dims_common) + [1]
SMALL_DNN_DIMS_COMMON = eval(FLAGS.small_dnn_hidden_dims_common) + [1]
BIAS_DIMS_COMMON = eval(FLAGS.bias_hidden_dims_common) + [1]

EME_DIM = FLAGS.embedding_dim # dense特征的embedding长度

NEED_TASK = FLAGS.need_task.split(",")

# print("dnn_common: {}\t bias_common: {}".format(DNN_DIMS_COMMON, BIAS_DIMS_COMMON))

TASK_NAME = NEED_TASK
TASK_NAME = ["is_cost_controllable"]
TASK_NAMES_DICT = {"is_cost_controllable": 2}
# TASK_NAMES_DICT = {"convert": 4, "cost": 5}
# TASK_NAME = ["convert","cost","convert_per_hour_label","cost_per_hour_label","convert_per_ad_hour_label","cost_per_ad_hour_label"]
# TASK_NAMES_DICT = {"convert": 0,"cost": 1,"convert_per_hour_label": 2,"cost_per_hour_label": 3,"convert_per_ad_hour_label": 4,"cost_per_ad_hour_label": 5}

def get_real_loss(labels, preds):
    return tf.square(labels-preds)

# features 输入的是tensorflow  tensor_dict
def supervised_model_fn(model, features, labels, mode, params, config):
    print("[qhz_debug] mode:", mode)
    logging_hook_di = dict()
    # logging_hook_di['[qhz_debug] labels'] = labels
    print("qhz_debug : labels", labels)
    # sparse.feature.FeatureSlot初始化设置，serving和training用户的特征是一样的
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

  
    # 特征字典，key为特征名称，value为分桶边界值
    sparse_features = SPARSE_FEAT_v2
    dense_features_1d = DENSE_FEAT_1D_v2
    dense_features_2d = DENSE_FEAT_2D_more

    # key为需要log化的特征名，value为log化后的分桶边界值
    need_log_feature = dict()
    if FLAGS.use_log_feature:
        need_log_feature = NEED_LOG_FEATURE_DICT

    # 三元组（'name_1', 'name_2', 'fc_name'），'fc_name'的特征 = 'name_1'特征 / 'name_2'特征
    ratio_list = RATIO_FEATURE
    # key为上述组合出的需要log化的'fc_name'特征名，value为log化后的分桶边界值
    ratio_feat_info = RATIO_FEATURE_INFO

    # key为需要delta操作的特征前缀名，value是特征的时长范围列表，有效的特征名={特征前缀名}_{时长范围}h_{特征类型}_all
    delta_camp = DELTA_CAMP_FEAT
    delta_camp_vmid = DELTA_CAMP_VMID_FEAT
    delta_vmid = DELTA_VMID_FEAT
    # key对上述有效的特征名在不同时间范围的均值下进行delta操作的delta特征，value为delta特征log化后的分桶边界值
    delta_feat_info = DELTA_FEATURE_INFO

    # DENSE_FEAT_2D_more二维特征中能找到camp和ad的特征映射关系的特征组合，后续对特征值做camp=camp-relu(ad)的操作，疑问：为什么只对2d做？为什么是这个操作？
    camp_to_ad = dict((k, CAMP_FEAT_TO_AD_FEAT_more[k]) for k in dense_features_2d.keys()
                   if k in CAMP_FEAT_TO_AD_FEAT_more)

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
    
    # 稀疏特征
    for feat_name, slot_id_hash_size in sorted(sparse_features.items()):
        slot_id, hash_size = slot_id_hash_size
        fs = model.add_feature_slot(slot_id, hash_size)
        if slot_id < 1024:
            fc = model.add_feature_column_v1(fs)
        else:  # fid v2
            fc = model.add_feature_column_v2(feat_name, fs)
        # 收集特征，累计维度
        if FLAGS.separate_embedding:
            # 分任务的embedding
            for t in TASK_NAME:
                state_embeddings_mt[t].append(fc.add_vector(sparse_emb_dim))
                lhuc_input_tensors_mt[t].append(fc.add_vector(sparse_emb_dim))
                cross_embeddings_mt[t].append(fc.add_vector(sparse_cross_emb_dim))
            cross_first_dim += sparse_cross_emb_dim
            lhuc_first_dim += sparse_emb_dim
        else:
            # 公共embedding
            common_emb = fc.add_vector(sparse_emb_dim)
            lhuc_common_emb = fc.add_vector(sparse_emb_dim)
            cross_common_emb = fc.add_vector(sparse_cross_emb_dim)
            for t in TASK_NAME:
                state_embeddings_mt[t].append(common_emb)
                lhuc_input_tensors_mt[t].append(lhuc_common_emb)
                cross_embeddings_mt[t].append(cross_common_emb)
            cross_first_dim += sparse_cross_emb_dim
            lhuc_first_dim += sparse_emb_dim
        # 收集特征名
        state_embeddings_names.append(feat_name)
    
    # 对camp_to_ad特征值做camp=camp-relu(ad)的操作
    # for k, v in camp_to_ad.items():
    #     features[k] = features[k] - tf.nn.relu(features[v])

    # dense特征分桶
    with tf.variable_scope("model_cur", reuse=tf.AUTO_REUSE,
                           partitioner=tf.fixed_size_partitioner(FLAGS.ps_num_embedding_shards, axis=0)):
        # 对log_dict(=NEED_LOG_FEATURE_DICT)特征做log(1+X)变化，对feat_2d(=CAMP_FEAT_HOURLY)特征做reduce_max变化，对所有变与不变的特征做分桶embedding
        dense_features_1d_embeddings, dense_features_1d_embedding_names, _, _, _ = bucket_feats(
            features, dense_features_1d, log_dict=need_log_feature, need_log1p=False, dim=EME_DIM,
            all_feat_suffix="common", feat_2d=CAMP_FEAT_HOURLY
        )
        # 收集特征名
        state_embeddings_names += dense_features_1d_embedding_names
        # 分任务收集特征，累计维度
        for t in TASK_NAME:
            state_embeddings_mt[t] += dense_features_1d_embeddings
            campaign_dense_embs_mt[t] += dense_features_1d_embeddings
            cross_embeddings_mt[t] +=  dense_features_1d_embeddings
        lhuc_first_dim += EME_DIM * len(dense_features_1d_embedding_names) #疑问：这里没收集lhuc的特征为什么加维度？
        campaign_dense_embs_size += EME_DIM * len(dense_features_1d_embedding_names)
        cross_first_dim += EME_DIM * len(dense_features_1d_embedding_names)

        # 对ratio_feat_list(=RATIO_FEATURE)做相除log变化，对delta_camp(=DELTA_CAMP_FEAT)和delta_vmid(=DELTA_VMID_FEAT)做相差log变化，对log_dict(=NEED_LOG_FEATURE_DICT)特征做log变化，
        dense_features_2d_embeddings, dense_features_2d_names, all_emb_size, \
        campaign_embeddings, campaign_vmid_embedding, vmid_emb_size = bucket_feats_2d(
            features, dense_features_2d, log_dict=need_log_feature, need_log1p=False, dim=EME_DIM, need_reduce="max",
            all_feat_suffix="common", seq_pooling=FLAGS.seq_pooling,
            ratio_feat_list=ratio_list, ratio_feat=ratio_feat_info,
            delta_camp=delta_camp,
            delta_camp_vmid=delta_camp_vmid, delta_vmid=delta_vmid, delta_feat=delta_feat_info,
            din=FLAGS.use_din, save_camp_vmid_hourly=camp_vmid_feat_hourly,
        )
        tmp_camp_emb_size = EME_DIM * len(campaign_embeddings)

        for t in TASK_NAME:
            state_embeddings_mt[t] += dense_features_2d_embeddings
            campaign_dense_embs_mt[t] += campaign_embeddings
            campaign_vmid_embs_mt[t].append(campaign_vmid_embedding)

        campaign_vmid_emb_size += vmid_emb_size
        campaign_dense_embs_size += tmp_camp_emb_size
        state_embeddings_names += dense_features_2d_names
        lhuc_first_dim += all_emb_size

    for t in TASK_NAME:
        assert len(state_embeddings_mt[t]) == len(
            state_embeddings_names), "len_state_embeddings_mt_{}: {} \t len_state_embeddings_names: {}".format(
            t, len(state_embeddings_mt[t]), len(state_embeddings_names)
        )
        for idx in range(len(state_embeddings_names)):
            tf.summary.histogram(
                'feat_{}_{}'.format(t, state_embeddings_names[idx]), state_embeddings_mt[t][idx]
            )
        print('[qhz_debug] state_embeddings_names:', state_embeddings_names)
        print('[qhz_debug] {} state_embeddings_mt: {}'.format(t, state_embeddings_mt[t]))

    # 对之前分任务的特征做汇总整合
    state_embedding_mt = dict()
    vmid_dense_embedding_mt = dict()
    campaign_dense_embedding_mt = dict()
    lhuc_embedding_mt = dict()
    cross_embedding_mt = dict()
    cross_embedding_merge_vmid_mt = dict()

    for t in TASK_NAME:
        state_embedding_mt[t] = tf.concat(state_embeddings_mt[t], axis=1)
        # print("task {}  State embedding dimensionality: {}".format(t, state_embedding_mt[t].get_shape()))
        # vmid_dense_embedding_mt[t] = campaign_vmid_embs_mt[t][-1]
        # merge_ = cross_embeddings_mt[t] + campaign_vmid_embs_mt[t]
        # cross_embedding_merge_vmid_mt[t] = tf.concat(merge_, axis=1)
        # campaign_dense_embedding_mt[t] = tf.concat(campaign_dense_embs_mt[t], axis=1)
        # cross_embedding_mt[t] = tf.concat(cross_embeddings_mt[t], axis=1)
        # print("task {}  cross embedding dimensionality: {} : first_dim: {}".format(
        #     t, cross_embedding_mt[t].get_shape(), cross_first_dim))
        # print("task {}  State embedding dimensionality: {}: lhuc_first_dim first dim: {}\n "
        #       "small lhuc: {}\t small lhuc emb_size: {}\t campaign_dense_embedding_mt: {}".format(
        #     t, state_embedding_mt[t].get_shape(), lhuc_first_dim,
        #     vmid_dense_embedding_mt[t].get_shape(), campaign_vmid_emb_size,
        #     campaign_dense_embedding_mt[t].get_shape()))
        lhuc_embedding_mt[t] = tf.concat(lhuc_input_tensors_mt[t], axis=1)
        # print("task {}  lhuc embedding dimensionality: {}".format(t, lhuc_embedding_mt[t].get_shape()))

    pred_mt = dict()
    with tf.variable_scope("model_cur", reuse=tf.AUTO_REUSE,
                           partitioner=tf.fixed_size_partitioner(FLAGS.ps_num_embedding_shards, axis=0)
                           ):
            # [batch, 1]
            logit_mt = list()
            logit_mt_dict = dict()
            is_train = (mode == tf.estimator.ModeKeys.TRAIN)
            # print("lhuc_first_dim: {} \t use_lhuc: {}", lhuc_first_dim, FLAGS.use_lhuc)
            for task in TASK_NAME:
                state_tensor = state_embedding_mt[task]
                lhuc_tensor = lhuc_embedding_mt[task]
                name = "all"

                logit_list = list()
                if FLAGS.use_lhuc:
                    print("qhz_debug: use lhuc")
                    # cost_pred
                    state_embedding_logit = get_lhuc_out(
                        state_tensor,
                        DNN_DIMS_COMMON, "uba_task_{}_score_{}".format(task, name),
                        lhuc_tensor,
                        first_dim=int(state_tensor.shape[1]),
                        concat_nn_input=False,
                        enable_bias=True,
                        lhuc_bottle_neck_dim=32,
                        is_train=is_train,
                        dropout_prob=FLAGS.dropout_prob,
                        task = task
                    )
                else:
                    print("qhz_debug: not use lhuc")
                    state_embedding_logit = get_dense_tower(
                        DNN_DIMS_COMMON,
                        state_tensor,
                        "uba_task_{}_score_{}".format(task, name),
                        is_train=is_train,
                        dropout_prob=FLAGS.dropout_prob
                    )

        
                state_embedding_logit = tf.reshape(state_embedding_logit, [-1])
                logit_list.append(state_embedding_logit)

                state_embedding_logit_merge = logit_list[0]
                logit_mt.append(state_embedding_logit_merge)
                logit_mt_dict[task] = state_embedding_logit_merge
                
                pred_mt[task] = state_embedding_logit     
    
    model.freeze_slots(features)
    # 添加模型参数
    if mode == tf.estimator.ModeKeys.PREDICT:
        '''
        assert len(logit_mt) > 0, "len_logit_mt <= 0"
        assert len(logit_mt_dict) > 0, "len_logit_mt_dict <= 0"
        zeros = tf.zeros_like(logit_mt[0], dtype=tf.float32)
        cost_w = get_pow_w("cost", tf.ones_like(logit_mt[0]), tf.ones_like(logit_mt[0]),
                           use_trans_learning=FLAGS.transfer_learning)
        active_w = get_pow_w("active", tf.ones_like(logit_mt[0]), tf.ones_like(logit_mt[0]),
                             use_trans_learning=FLAGS.transfer_learning)
        convert_w = get_pow_w("convert", tf.ones_like(logit_mt[0]), tf.ones_like(logit_mt[0]),
                              use_trans_learning=FLAGS.transfer_learning)

        if FLAGS.use_recify:
            logit_cost = logit_mt_dict.get("cost", zeros) - tf.math.log(cost_w)
            logit_active = logit_mt_dict.get("active", zeros) - tf.math.log(active_w)
            logit_convert = logit_mt_dict.get("convert", zeros) - tf.math.log(convert_w)
        else:
            logit_cost = logit_mt_dict.get("cost", zeros)
            logit_active = logit_mt_dict.get("active", zeros)
            logit_convert = logit_mt_dict.get("convert", zeros)

        predictions_dict = {
            "cost": tf.nn.sigmoid(logit_cost),
            "active": tf.nn.sigmoid(logit_active),
            "convert": tf.nn.sigmoid(logit_convert)
        }
        '''
        predictions_dict = {}
        for task in TASK_NAME:
            predictions_dict[task] = pred_mt[task]
        predictions_dict['is_cost_controllable'] = pred_mt['is_cost_controllable']
                   
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
            if FLAGS.loss_type == 'mae':
                loss_pre = tf.abs(task_label - task_score)
                loss = tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss:mae")
            elif FLAGS.loss_type == 'huber':
                loss_pre = huber_loss(task_label, task_score, FLAGS.huber_delta)
                loss = tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss:huber_loss")
            elif FLAGS.loss_type == 'weighted_cross_entropy':
                loss_pre = weighted_cross_entropy(task_label, task_score, FLAGS.cross_entropy_weight, logging_hook_di)
                loss = tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss:weighted_cross_entropy")
            elif FLAGS.loss_type == 'cross_entropy':
                loss_pre = cross_entropy(task_label, task_score, logging_hook_di)
                loss = tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss:cross_entropy")
            else:
                loss_pre = get_real_loss(task_label, task_score)
                loss = tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss: mse")
            logging_hook_di['[qhz_debug] reduce_mean_loss'] = loss
            loss_mt[task] = loss
            loss_merge += loss
            logging_hook_di['[qhz_debug] loss_merge'] = loss_merge
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
            
            di2_multihead_hook = train.DeepInsight2MultiHeadHook(
                tf.reshape(mock_user, [-1]),
                tf.reshape(req_time_tensor, [-1]),
                score_tensor_dict=save_scores_tensor_dict,
                label_tensor_dict=save_labels_tensor_dict,
                extra_tensors={
                    'dataset': tf.fill(tf.shape(logit_mt[0]), 'eval'),
                    'req_id': tf.reshape(features['req_id'], [-1]),
                    'customer_id': tf.reshape(features['customer_id'], [-1]),
                    'advertiser_id': tf.reshape(features['advertiser_id'], [-1]),
                    # 'campaign_id': tf.reshape(features['campaign_id'], [-1]),
                    'external_action': tf.reshape(features['external_action'], [-1]),
                    'deep_external_action': tf.reshape(features['deep_external_action'], [-1]),
                    'deep_bid_type': tf.reshape(features['deep_bid_type'], [-1]),
                    'app_package': tf.reshape(features['app_package'], [-1]),
                    'target_app_package': tf.reshape(features['target_app_package'], [-1]),
                    'fc_real_cost_72h_vmid_all': tf.reshape(features['fc_real_cost_72h_vmid_all'], [-1]),
                    'fc_pvr_72h_vmid_all': tf.reshape(features['fc_pvr_72h_vmid_all'], [-1]),
                    'cost_label': tf.reshape(labels[:, 0], [-1]),
                    'convert_label': tf.reshape(labels[:, 1], [-1]),
                    'advv_label': tf.reshape(labels[:, 2], [-1]),
                    'sum_pvr_label': tf.reshape(labels[:, 3], [-1]),
                    'cost_controllable_label': tf.reshape(labels[:, 4], [-1]),
                    'nobid_cost_label': tf.reshape(labels[:, 5], [-1]),
                    'p_date': tf.reshape(labels[:, 6], [-1]),

                },
                neg_sample_rate=1.0
            )
            update_metric_tensors = dict()
            for k, v in eval_tensors.items():
                update_metric_tensors[k] = tf.metrics.mean(v)

            for task_name in TASK_NAME:
                update_metric_tensors["acc_{}".format(task_name)] = tf.metrics.accuracy(
                    labels=save_labels_tensor_dict[task_name],
                    predictions=save_scores_tensor_dict[task_name],
                    name='acc_op_{}'.format(task_name))

                update_metric_tensors["auc_{}".format(task_name)] = tf.metrics.auc(
                    labels=save_labels_tensor_dict[task_name],
                    predictions=save_scores_tensor_dict[task_name],
                    weights=None, num_thresholds=200, metrics_collections=None,
                    updates_collections=None, curve='ROC',
                    summation_method='trapezoidal', thresholds=None,
                    name='auc_op_{}'.format(task_name))

                # 这边是在通过auc函数来算recall吗？
                update_metric_tensors["recall_{}".format(task_name)] = tf.metrics.auc(
                    labels=save_labels_tensor_dict[task_name],
                    predictions=save_scores_tensor_dict[task_name],
                    thresholds=[0.08, 0.1, 0.2, 0.5],
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
                logit_post = logit_post_mt_dict[task_name]
                score = score_mt_dict[task_name]
                is_auto_type = tf.reshape(is_auto_type, [-1])

                

                save_scores_tensor_dict['class_{}'.format(task_name)] = tf.reshape(score, [-1])
                save_scores_tensor_dict['regression_{}'.format(task_name)] = tf.reshape(logit_post, [-1])
                save_scores_tensor_dict['is_eval_{}'.format(task_name)] = tf.reshape(
                    tf.zeros_like(logit_post, tf.float32),
                    [-1]
                )

                save_labels_tensor_dict['class_{}'.format(task_name)] = tf.reshape(label_mt_dict[task_name], [-1])
                save_labels_tensor_dict['regression_{}'.format(task_name)] = tf.reshape(value_mt_dict[task_name], [-1])
                save_labels_tensor_dict['is_eval_{}'.format(task_name)] = tf.reshape(
                    tf.zeros_like(logit_pre, tf.float32),
                    [-1]
                )
                '''

            di2_multihead_hook = train.DeepInsight2MultiHeadHook(
                tf.reshape(mock_user, [-1]),
                tf.reshape(req_time_tensor, [-1]),
                score_tensor_dict=save_scores_tensor_dict,
                label_tensor_dict=save_labels_tensor_dict,
                extra_tensors={
                    'dataset': tf.fill(tf.shape(logit_mt[0]), 'train'),
                    'req_id': tf.reshape(features['req_id'], [-1]),
                    'customer_id': tf.reshape(features['customer_id'], [-1]),
                    'advertiser_id': tf.reshape(features['advertiser_id'], [-1]),
                    # 'campaign_id': tf.reshape(features['campaign_id'], [-1]),
                    'external_action': tf.reshape(features['external_action'], [-1]),
                    'deep_external_action': tf.reshape(features['deep_external_action'], [-1]),
                    'deep_bid_type': tf.reshape(features['deep_bid_type'], [-1]),
                    'app_package': tf.reshape(features['app_package'], [-1]),
                    'target_app_package': tf.reshape(features['target_app_package'], [-1]),
                    'fc_real_cost_72h_vmid_all': tf.reshape(features['fc_real_cost_72h_vmid_all'], [-1]),
                    'fc_pvr_72h_vmid_all': tf.reshape(features['fc_pvr_72h_vmid_all'], [-1]),
                    'cost_label': tf.reshape(labels[:, 0], [-1]),
                    'convert_label': tf.reshape(labels[:, 1], [-1]),
                    'advv_label': tf.reshape(labels[:, 2], [-1]),
                    'sum_pvr_label': tf.reshape(labels[:, 3], [-1]),
                    'cost_controllable_label': tf.reshape(labels[:, 4], [-1]),
                    'nobid_cost_label': tf.reshape(labels[:, 5], [-1]),
                    'p_date': tf.reshape(labels[:, 6], [-1]),
                },
                neg_sample_rate=1.0
            )
            main_opt = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate)
            main_vars = get_vars_not_in_scope("model_bias")
            main_grads_and_vars = main_opt.compute_gradients(loss=loss_merge, var_list=main_vars)
            for v, (grad, val) in zip(main_vars, main_grads_and_vars):
                if grad is not None:
                    tf.summary.histogram("grad_{}".format(v), grad)
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
    checkpoint_dir = os.path.join(FLAGS.model_path, 'checkpoints')
    if FLAGS.last_model_path and not tf.train.latest_checkpoint(checkpoint_dir):
        if FLAGS.batch_reload:
            warmup_dir = os.path.join(FLAGS.last_model_path, 'checkpoints')
            print("input last_model_path: {} \t do reload!".format(checkpoint_dir))
        else:
            warmup_dir = None
            print("input last_model_path: {} \t but do not reload!".format(checkpoint_dir))
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
    model, run_config = get_estimator()

    sparse_features = SPARSE_FEAT_v2
    dense_features_1d = DENSE_FEAT_1D_v2
    dense_features_2d = DENSE_FEAT_2D_more

    feat_2d = dict((k, CAMP_FEAT_TO_AD_FEAT_more[k]) for k in dense_features_2d.keys()
                       if k in CAMP_FEAT_TO_AD_FEAT_more)

    dense_features_1d_real_2d = [k for k in dense_features_1d.keys() if k in CAMP_FEAT_HOURLY]
    dense_features_1d_real_1d = [k for k in dense_features_1d.keys() if k not in CAMP_FEAT_HOURLY]

    dense_features = {fc: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=[0]) for fc in
                      sorted(dense_features_1d_real_1d + EXTRA_FEAT)}

    dense_features.update({fc: tf.io.FixedLenFeature(shape=[20], dtype=tf.int64, default_value=[0] * 20) for fc in
                           sorted(list(dense_features_2d.keys())) + list(feat_2d.values()) + dense_features_1d_real_2d})

    # dense_features.update({fc: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32, default_value=[0.0]) for fc in
    #                        sorted(list(NAME_TO_LABEL.values()))})

    num_shards = run_config.num_worker_replicas
    shard_id = run_config.task_id + (0 if run_config.is_chief else 1)

    if FLAGS.is_train == 1:
        print('Training start!')
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
        print('Training finishes!')

    if FLAGS.is_eval == 1:
        print('Evaluation starts!')
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
        print('eval metric_ops = {}'.format(eval_metrics))
        print('Evaluation finishes!')

    return model, run_config


def serving_input_receiver_fn():
    dense_features_1d = DENSE_FEAT_1D_v2
    dense_features_2d = DENSE_FEAT_2D_more

    feat_2d = dict((k, CAMP_FEAT_TO_AD_FEAT_more[k]) for k in dense_features_2d.keys() if k in CAMP_FEAT_TO_AD_FEAT_more)

    features_next = {
        'fids_indices': tf.placeholder(tf.int64, shape=[None], name='fids_indices'),
        'fids_values': tf.placeholder(tf.int64, shape=[None], name='fids_values'),
        'fids_dense_shape': tf.placeholder(tf.int64, shape=[None], name='fids_dense_shape')
    }
    dense_features_1d_real_2d = [k for k in dense_features_1d.keys() if k in CAMP_FEAT_HOURLY]
    dense_features_1d_real_1d = [k for k in dense_features_1d.keys() if k not in CAMP_FEAT_HOURLY]
    features_next.update({fc: tf.placeholder(dtype=tf.int64, shape=[None, 20], name=fc) for fc in \
                          sorted(list(dense_features_2d.keys()) + list(feat_2d.values()) + dense_features_1d_real_2d)})
    features_next.update({fc: tf.placeholder(dtype=tf.int64, shape=[None, 1], name=fc) for fc in \
                          sorted(dense_features_1d_real_1d + EXTRA_FEAT)})
    print("qhz_debug : features_next", features_next)
    return tf.estimator.export.ServingInputReceiver(features_next, features_next)


def run(_):
    model, _ = train_and_evaluate()
    if model.config.is_chief and FLAGS.is_train == 1:
        # TODO: check下path有没有问题
        model.export_saved_model(
            export_dir_base=os.path.join(FLAGS.model_path, 'exported'),
            serving_input_receiver_fn=serving_input_receiver_fn
        )


if __name__ == '__main__':
    tf.app.run(run)
