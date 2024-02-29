# BB-8 and R2-D2 are best friends.

import sys
sys.path.insert(0, '../FM')
sys.path.insert(0, '../lastfm')

import pickle
import torch
import argparse

import time
import numpy as np

from config import global_config as cfg
from epi import run_one_episode, update_PN_model
from pn import PolicyNetwork
import copy
import random
import json

from collections import defaultdict

the_max = 0
for k, v in cfg.item_dict.items():
    if the_max < max(v['feature_index']):
        the_max = max(v['feature_index'])
print('The max is: {}'.format(the_max))
FEATURE_COUNT = the_max + 1


def cuda_(var):
    return var.cuda() if torch.cuda.is_available()else var


def main():
    # -eval 0 -initeval 0 -trick 0 -mini 1 -alwaysupdate 1 -upcount 1 -upreg 0.001 -code stable -mask 0 -purpose train -mod ear 
    parser = argparse.ArgumentParser(description="Run conversational recommendation.")
    parser.add_argument('-mt', type=int, default=15, dest='mt', help='MAX_TURN') 
    parser.add_argument('-playby', type=str, default='policy', dest='playby', help='playby')
    # options include:
    # AO: (Ask Only and recommend by probability)
    # RO: (Recommend Only)
    # policy: (action decided by our policy network)
    parser.add_argument('-fmCommand', type=str, default=8, dest='fmCommand', help='fmCommand')
    # the command used for FM, check out /EAR/lastfm/FM/
    parser.add_argument('-optim', type=str, default='SGD', dest='optim', help='optimizer')
    # the optimizer for policy network
    parser.add_argument('-lr', type=float, default=0.0001, dest='lr', help='lr')
    # learning rate of policy network
    parser.add_argument('-decay', type=float, default=0, dest='decay', help='decay')
    # weight decay
    parser.add_argument('-TopKTaxo', type=int, default=3, dest='TopKTaxo', help='TopKTaxo')
    # how many 2-layer feature will represent a big feature. Only Yelp dataset use this param, lastFM have no effect.
    parser.add_argument('-gamma', type=float, default=0.1, dest='gamma', help='gamma')
    # gamma of training policy network
    parser.add_argument('-trick', type=int, default=0, dest='trick', help='trick')
    # whether use normalization in training policy network
    parser.add_argument('-startFrom', type=int, default=0, dest='startFrom', help='startFrom')
    # startFrom which user-item interaction pair
    parser.add_argument('-endAt', type=int, default=11891, dest='endAt', help='endAt')
    # endAt which user-item interaction pair
    parser.add_argument('-strategy', type=str, default='maxent', dest='strategy', help='strategy')
    # strategy to choose question to ask, only have effect
    parser.add_argument('-eval', type=int, default=0, dest='eval', help='eval')
    # whether current run is for evaluation
    parser.add_argument('-mini', type=int, default=0, dest='mini', help='mini')
    # means `mini`-batch update the FM
    parser.add_argument('-alwaysupdate', type=int, default=1, dest='alwaysupdate', help='alwaysupdate')
    # means always mini-batch update the FM, alternative is that only do the update for 1 time in a session.
    # we leave this exploration tof follower of our work.
    parser.add_argument('-initeval', type=int, default=0, dest='initeval', help='initeval')
    # whether do the evaluation for the `init`ial version of policy network (directly after pre-train)
    parser.add_argument('-upoptim', type=str, default='SGD', dest='upoptim', help='upoptim')
    # optimizer for reflection stafe
    parser.add_argument('-upcount', type=int, default=1, dest='upcount', help='upcount')
    # how many times to do reflection
    parser.add_argument('-upreg', type=float, default=0.001, dest='upreg', help='upreg')
    # regularization term in
    parser.add_argument('-code', type=str, default='stable', dest='code', help='code')
    # We use it to give each run a unique identifier.
    parser.add_argument('-purpose', type=str, default='train', dest='purpose', help='purpose')
    # options: pretrain, others
    parser.add_argument('-mod', type=str, default='crm', dest='mod', help='mod')
    # options: CRM, EAR
    parser.add_argument('-mask', type=int, default=0, dest='mask', help='mask')
    # use for ablation study, 1, 2, 3, 4,5,6,7
    parser.add_argument('-interact', type=int, default=1, metavar='<interact>', dest='interact', help='interact')
    # interact method, 0 无attribute 作为输入

    A = parser.parse_args()
    dataset='LastFM'
    f_result= open(f'../../result/FM/{dataset}_{A.mod}_startFrom_{A.startFrom}_endAt_{A.endAt}_mask_{A.mask}_eval_{A.eval}_interact_{A.interact}_result.csv', 'w', encoding='utf-8')
    f_result.write('Success_Turn,recall_1,recall_5,recall_10,MRR_1,MRR_5,MRR_10' + '\n')
    f_result.flush()
    cfg.change_param(playby=A.playby, eval=A.eval, update_count=A.upcount, update_reg=A.upreg, purpose=A.purpose, mod=A.mod, mask=A.mask, interact=A.interact)

    random.seed(1)

    # we random shuffle and split the valid and test set, for Action Stage training and evaluation respectively, to avoid the bias in the dataset.
    all_list = cfg.valid_list + cfg.test_list
    print('The length of all list is: {}'.format(len(all_list)))
    random.shuffle(all_list)
    the_valid_list = all_list[: int(len(all_list) / 2.0)]
    the_test_list = all_list[int(len(all_list) / 2.0):]

    gamma = A.gamma
    FM_model = cfg.FM_model

    if A.eval == 1:
        if A.mod == 'ear':
            if A.interact == 0:
                fp = f'../../../data/PN-model-ear/PN-model-ear-mask-{A.mask}-interact0.txt'
            else:
                fp = f'../../../data/PN-model-ear/PN-model-ear-mask-{A.mask}.txt'
        if A.mod == 'crm':
            if A.interact == 0:
                fp = f'../../../data/PN-model-crm/PN-model-crm-mask-{A.mask}-interact0.txt'
            else:
                fp = f'../../../data/PN-model-crm/PN-model-crm-mask-{A.mask}.txt'
        if A.initeval == 1:
            if A.mod == 'ear':
                if A.interact == 0:
                    fp = '../../../data/PN-model-ear/pretrain-model-interact0.pt'
                else:
                    fp = '../../../data/PN-model-ear/pretrain-model.pt'
            if A.mod == 'crm':
                if A.interact == 0:
                    fp = '../../../data/PN-model-crm/pretrain-model-interact0.pt'
                else:
                    fp = '../../../data/PN-model-crm/pretrain-model.pt'
    else:
        # means training
        if A.mod == 'ear':
            if A.interact == 0:
                fp = '../../../data/PN-model-ear/pretrain-model-interact0.pt'
            else:
                fp = '../../../data/PN-model-ear/pretrain-model.pt'
        if A.mod == 'crm':
            if A.interact == 0:
                fp = '../../../data/PN-model-crm/pretrain-model-interact0.pt'
            else:
                fp = '../../../data/PN-model-crm/pretrain-model.pt'
    INPUT_DIM = 0
    if A.mod == 'ear':
        # INPUT_DIM = 89 #  s_ent + s_pre + s_his + s_len 
        INPUT_DIM = 126 #  s_ent + s_pre + s_his + s_len + s_seq #TODO: all state
    if A.mod == 'crm':
        INPUT_DIM = 126 # s_seq
        # INPUT_DIM = 126 #  s_seq + s_ent + s_pre + s_his + s_len #TODO: all state
    print('fp is: {}'.format(fp))
    PN_model = PolicyNetwork(input_dim=INPUT_DIM, dim1=64, output_dim=34)
    start = time.time()

    try:
        PN_model.load_state_dict(torch.load(fp))
        print('Now Load PN pretrain from {}, takes {} seconds.'.format(fp, time.time() - start))
    except:
        print('Cannot load the model!!!!!!!!!\n fp is: {}'.format(fp))
        if A.playby == 'policy':
            sys.exit()

    if A.optim == 'Adam':
        optimizer = torch.optim.Adam(PN_model.parameters(), lr=A.lr, weight_decay=A.decay)
    if A.optim == 'SGD':
        optimizer = torch.optim.SGD(PN_model.parameters(), lr=A.lr, weight_decay=A.decay)
    if A.optim == 'RMS':
        optimizer = torch.optim.RMSprop(PN_model.parameters(), lr=A.lr, weight_decay=A.decay)

    numpy_list = list()
    NUMPY_COUNT = 0

    sample_dict = defaultdict(list)
    conversation_length_list = list()
    for epi_count in range(A.startFrom, A.endAt):
        if epi_count % 1 == 0:
            print('-----\nIt has processed {} episodes'.format(epi_count))
        start = time.time()

        if A.purpose != 'pretrain':
            u, item = the_valid_list[epi_count]

        # if A.test == 1 or A.eval == 1:
        if A.eval == 1:
            u, item = the_test_list[epi_count]

        if A.purpose == 'fmdata':
            u, item = 0, epi_count

        if A.purpose == 'pretrain':
            u, item = cfg.train_list[epi_count]

        current_FM_model = copy.deepcopy(FM_model)
        param1, param2 = list(), list()
        param3 = list()
        param4 = list()
        i = 0
        for name, param in current_FM_model.named_parameters():
            param4.append(param)
            # print(name, param)
            if i == 0:
                param1.append(param)
            else:
                param2.append(param)
            if i == 2:
                param3.append(param)
            i += 1
        optimizer1_fm = torch.optim.Adagrad(param1, lr=0.01, weight_decay=A.decay)
        optimizer2_fm = torch.optim.SGD(param4, lr=0.001, weight_decay=A.decay)
        optimizer3 = torch.optim.SGD(param3, lr=0.001, weight_decay=A.decay)# TODO: reject attribute online update
        user_id = int(u)
        item_id = int(item)

        # write_fp = '../../../data/interaction-log/{}/v4-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}-m-{}.txt'.format(
        write_fp = '../../../data/interaction-log/{}/v4-s-{}-e-{}-gamma-{}-playby-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-m-{}-interact-{}.txt'.format(
            A.mod.lower(), A.startFrom, A.endAt,A.gamma, A.playby,
            A.eval, A.initeval,
            A.mini, A.alwaysupdate, A.upcount, A.mask, A.interact)

        choose_pool = cfg.item_dict[str(item_id)]['categories']

        if A.purpose not in ['pretrain', 'fmdata']:
            # this means that: we are not collecting data for pretraining or fm data
            # then we only randomly choose one start attribute to ask!
            choose_pool = [random.choice(choose_pool)]

        for c in choose_pool:
            with open(write_fp, 'a') as f:
                f.write(
                    'Starting new\nuser ID: {}, item ID: {} episode count: {}, feature: {}\n'.format(user_id, item_id, epi_count, cfg.item_dict[str(item_id)]['categories']))
            start_facet = c
            if A.purpose != 'pretrain': #log_prob_list: (1,current_turn)
                log_prob_list, rewards, rec_success_record = run_one_episode(current_FM_model, user_id, item_id, A.mt, False, write_fp,
                                                         A.strategy, A.TopKTaxo,
                                                         PN_model, gamma, A.trick, A.mini,
                                                         optimizer1_fm, optimizer2_fm, optimizer3, A.alwaysupdate, start_facet,
                                                         A.mask, sample_dict)# TODO: reject attribute online update
                if rec_success_record is None:
                    continue
                else:
                    rec_success_record = [f'{mt:.4f}' for mt in rec_success_record]
                    line = ','.join(rec_success_record) + '\n'
                    f_result.write(line)
                    f_result.flush()
            else:
                current_np = run_one_episode(current_FM_model, user_id, item_id, A.mt, False, write_fp,
                                                         A.strategy, A.TopKTaxo,
                                                         PN_model, gamma, A.trick, A.mini,
                                                         optimizer1_fm, optimizer2_fm, optimizer3, A.alwaysupdate, start_facet,
                                                         A.mask, sample_dict)# TODO: reject attribute online update
                numpy_list += current_np #numpy_list: ((action, state_vector))

            # update PN model
            if A.playby == 'policy' and A.eval != 1:
                update_PN_model(PN_model, log_prob_list, rewards, optimizer)
                print('updated PN model')
                current_length = len(log_prob_list)
                conversation_length_list.append(current_length)
            # end update

            if A.purpose != 'pretrain':
                with open(write_fp, 'a') as f:
                    f.write('Big features are: {}\n'.format(choose_pool)) # GT attr
                    if rewards is not None:
                        f.write('reward is: {}\n'.format(rewards.data.numpy().tolist()))
                    f.write('WHOLE PROCESS TAKES: {} SECONDS\n'.format(time.time() - start))

        # Write to pretrain numpy.
        if A.purpose == 'pretrain':
            if len(numpy_list) > 5000:
                with open('../../../data/pretrain-numpy-data-{}-FM-interact-{}/segment-{}-start-{}-end-{}.pk'.format(
                        A.mod, A.interact, NUMPY_COUNT, A.startFrom, A.endAt), 'wb') as f:
                    pickle.dump(numpy_list, f)
                    print('Have written 5000 numpy arrays!')
                NUMPY_COUNT += 1
                numpy_list = list()
        # numpy_list is a list of list.
        # e.g. numpy_list[0][0]: int, indicating the action.
        # numpy_list[0][1]: state,   a one-d array of length 89 for EAR, and 33 for CRM.
        # end write

        # Write sample dict:
        if A.purpose == 'fmdata' and A.playby != 'AOO_valid':
            if epi_count % 100 == 1:
                with open('../../../data/sample-dict/start-{}-end-{}.json'.format(A.startFrom, A.endAt), 'w') as f:
                    json.dump(sample_dict, f, indent=4)
        # end write
        if A.purpose == 'fmdata' and A.playby == 'AOO_valid':
            if epi_count % 100 == 1:
                with open('../../../data/sample-dict/valid-start-{}-end-{}.json'.format(A.startFrom, A.endAt),
                          'w') as f:
                    json.dump(sample_dict, f, indent=4)

        check_span = 500
        if epi_count % check_span == 0 and epi_count >= 3 * check_span and cfg.eval != 1 and A.purpose != 'pretrain':
            # We use AT (average turn of conversation) as our stopping criterion
            # in training mode, save RL model periodically
            # save model first
            # PATH = '../../../data/PN-model-{}/v4-s-{}-e-{}-gamma-{}-playby-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-m-{}-interact-{}-epi-{}.txt'.format(
            #     A.mod.lower(), A.startFrom, A.endAt, A.gamma,A.playby,
            #     A.eval, A.initeval,
            #     A.mini, A.alwaysupdate, A.upcount, A.mask, A.interact, epi_count)
            if A.interact == 0:
                PATH = '../../../data/PN-model-{}/PN-model-{}-mask-{}-interact0.txt'.format(A.mod.lower(), A.mod.lower(), A.mask)
            else: 
                PATH = '../../../data/PN-model-{}/PN-model-{}-mask-{}.txt'.format(A.mod.lower(), A.mod.lower(), A.mask)
            # PATH = '../../../data/PN-model-{}/v4-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}-m-{}-epi-{}.txt'.format(
            #     A.mod.lower(), A.code, A.startFrom, A.endAt, A.lr, A.gamma, A.playby, A.strategy, A.TopKTaxo, A.trick,
            #     A.eval, A.initeval,
            #     A.mini, A.alwaysupdate, A.upcount, A.upreg, A.mask, epi_count)
            torch.save(PN_model.state_dict(), PATH)
            print('Model saved at {}'.format(PATH))

            # a0 = conversation_length_list[epi_count - 4 * check_span: epi_count - 3 * check_span]
            #每500轮算一次AT average turn
            a1 = conversation_length_list[epi_count - 3 * check_span: epi_count - 2 * check_span]
            a2 = conversation_length_list[epi_count - 2 * check_span: epi_count - 1 * check_span]
            a3 = conversation_length_list[epi_count - 1 * check_span: ]
            a1 = np.mean(np.array(a1))
            a2 = np.mean(np.array(a2))
            a3 = np.mean(np.array(a3))

            with open(write_fp, 'a') as f:
                f.write('$$$current turn: {}, a3: {}, a2: {}, a1: {}\n'.format(epi_count, a3, a2, a1))
            print('current turn: {}, a3: {}, a2: {}, a1: {}'.format(epi_count, a3, a2, a1))

            # num_interval = int(epi_count / check_span)
            # for i in range(num_interval):
            #     ave = np.mean(np.array(conversation_length_list[i * check_span: (i + 1) * check_span]))
            #     print('start: {}, end: {}, average: {}'.format(i * check_span, (i + 1) * check_span, ave))
            #     PATH = '../../../data/PN-model-{}/v4-s-{}-e-{}-gamma-{}-playby-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-m-{}-interact-{}-epi-{}.txt'.format(
            #         A.mod.lower(), A.startFrom, A.endAt, A.gamma, A.playby, 
            #         A.eval, A.initeval,
            #         A.mini, A.alwaysupdate, A.upcount, A.mask, A.interact, (i + 1) * check_span)
            #     # PATH = '../../../data/PN-model-{}/v4-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}-m-{}-epi-{}.txt'.format(
            #     #     A.mod.lower(), A.code, A.startFrom, A.endAt, A.lr, A.gamma, A.playby, A.strategy, A.TopKTaxo,
            #     #     A.trick,
            #     #     A.eval, A.initeval,
            #     #     A.mini, A.alwaysupdate, A.upcount, A.upreg, A.mask, (i + 1) * check_span)
            #     print('Model saved at: {}'.format(PATH))

            if a3 > a1 and a3 > a2:
                print('Early stop of RL!')
                exit()
    f_result.close()

if __name__ == '__main__':
    main()
