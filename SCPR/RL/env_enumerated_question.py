import json
import numpy as np
import itertools
import os
import random
from utils import *

from tkinter import _flatten
from collections import Counter
class EnumeratedRecommendEnv(object):
    def __init__(self, kg, dataset, data_name, seed=1, max_turn=15, cand_len_size=20, attr_num=20, mode='train', command=6, ask_num=1, entropy_way='weight entropy', fm_epoch=0, reward_pre=0):
        self.data_name = data_name
        self.command = command
        self.mode = mode
        self.seed = seed
        self.max_turn = max_turn    
        self.reject_feature_embeds_size = 64
        self.reject_item_embeds_size = 64
        self.user_rej_item=[]
        self.attr_state_num = attr_num
        self.cand_len_size = cand_len_size
        self.kg = kg
        self.dataset = dataset
        self.reward_pre = reward_pre
        self.feature_length = getattr(self.dataset, 'large_feature').value_len
        self.user_length = getattr(self.dataset, 'user').value_len
        self.item_length = getattr(self.dataset, 'item').value_len

        # action parameters
        self.ask_num = ask_num
        self.rec_num = 10
        self.target_item_position=-1
        #  entropy  or weight entropy
        self.ent_way = entropy_way

        # user's profile
        self.reachable_feature = []   # user reachable large_feature
        self.user_acc_feature = []  # user accepted large_feature which asked by agent
        self.user_rej_feature = []  # user rejected large_feature which asked by agent
        self.acc_samll_fea = []
        self.rej_samll_fea = []
        self.cand_items = []   # candidate items


        #user_id  item_id   cur_step   cur_node_set
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0        #  
        self.cur_node_set = []
        # state veactor
        self.user_embed = None
        self.user_embed_v2 = None
        self.item_embed = None
        self.reject_feature_embed = None
        self.reject_item_embed = None
        self.reject_item_embed_mean = None
        self.reject_feature_embed_mean = None
        self.conver_his = []    #conversation_history
        self.cand_len = []    #the number of candidate items  [ binary]
        self.attr_ent = []  # attribute entropy
        self.one_hot=[0] * 29
        self.one_hot_encoder_yelp=29
        self.ui_embeds_v2_size=4
        self.ui_dict = self.__load_rl_data__(data_name, mode=mode)  # np.array [ u i weight]
        self.user_weight_dict = dict()
        self.user_items_dict = dict()

        #init seed & init user_dict
        set_random_seed(self.seed) # set random seed
        if mode == 'train':
            self.__user_dict_init__() # init self.user_weight_dict  and  self.user_items_dict
        elif mode == 'test':
            self.ui_array = None    # u-i array [ [userID1, itemID1], ..., [userID2, itemID2]]
            self.__test_tuple_generate__()
            self.test_num = 0
        # embeds = {
        #     'ui_emb': ui_emb,
        #     'feature_emb': feature_emb
        # }
        #load fm epoch
        embeds = load_embed(data_name, epoch=fm_epoch)
        self.ui_embeds =embeds['ui_emb']
        self.feature_emb = embeds['feature_emb']
        # self.feature_length = self.feature_emb.shape[0]-1

        self.action_space = 2

        self.state_space_dict = {
            1: self.max_turn + self.cand_len_size + self.attr_state_num + self.ui_embeds.shape[1],
            2: self.attr_state_num,  # attr_ent
            3: self.max_turn,  #conver_his
            4: self.cand_len_size,  #cand_item
            5: self.cand_len_size + self.max_turn + self.ui_embeds_v2_size, # user_embedding is same to CRIF
            6: self.cand_len_size + self.attr_state_num + self.max_turn, #attr_ent + conver_his + cand_item
            7: self.cand_len_size + self.max_turn,
            8: self.cand_len_size + self.max_turn + self.one_hot_encoder_yelp,
            9: self.cand_len_size + self.max_turn + self.ui_embeds.shape[1],
            10: self.cand_len_size + self.max_turn + self.reject_feature_embeds_size+ self.reject_item_embeds_size

        }
        self.state_space = self.state_space_dict[self.command]
        self.prev_reward = - 0.01
        self.reward_dict_add_reward_pre = {
            'ask_suc': 0.01+self.prev_reward,
            'ask_fail': -0.1+self.prev_reward,
            'rec_suc': 1+self.prev_reward,
            'rec_fail': -0.1+self.prev_reward,
            'until_T': -0.3,     
            'cand_none': -0.1
        }
        self.reward_dict = {
            'ask_suc': 0.01,
            'ask_fail': -0.1,
            'rec_suc': 1,
            'rec_fail': -0.1,
            'until_T': -0.3,      # MAX_Turn
            'cand_none': -0.1
        }
        self.history_dict = {
            'ask_suc': 1,
            'ask_fail': -1,
            'rec_scu': 2,
            'rec_fail': -2,
            'until_T': 0
        }
        self.attr_count_dict = dict()   # This dict is used to calculate entropy

    def __load_rl_data__(self, data_name, mode):
        if mode == 'train':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_valid.json'), encoding='utf-8') as f:
                print('train_data: load RL valid data')
                mydict = json.load(f)
        elif mode == 'test':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_test.json'), encoding='utf-8') as f:
                mydict = json.load(f)
        return mydict


    def __user_dict_init__(self):   #Calculate the weight of the number of interactions per user
        ui_nums = 0
        for items in self.ui_dict.values():
            ui_nums += len(items)
        for user_str in self.ui_dict.keys():
            user_id = int(user_str)
            self.user_weight_dict[user_id] = len(self.ui_dict[user_str])/ui_nums
        print('user_dict init successfully!')

    def __test_tuple_generate__(self):
        ui_list = []
        for user_str, items in self.ui_dict.items():
            user_id = int(user_str)
            for item_id in items:
                ui_list.append([user_id, item_id])
        self.ui_array = np.array(ui_list)
        np.random.shuffle(self.ui_array)

    def reset(self):
        #init  user_id  item_id
        self.cur_conver_step = 0   #reset cur_conversation step
        if self.mode == 'train':
            users = list(self.user_weight_dict.keys())
            #TODO select user by weight?
            #self.user_id = np.random.choice(users, p=list(self.user_weight_dict.values())) # select user  according to user weights
            self.user_id = np.random.choice(users)
            self.target_item = np.random.choice(self.ui_dict[str(self.user_id)])
        elif self.mode == 'test':
            self.user_id = self.ui_array[self.test_num, 0]
            self.target_item = self.ui_array[self.test_num, 1]
            self.test_num += 1
        # init user's profile
        print('-----------reset state vector------------')
        self.user_acc_feature = []  # user accepted large_feature which asked by agent
        self.user_rej_feature = []  # user rejected large_feature which asked by agent
        self.acc_samll_fea = []
        self.rej_samll_fea = []
        self.user_rej_item = []
        self.cand_items = list(range(self.item_length))
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))

        # init  state vector
        self.user_embed = self.ui_embeds[self.user_id].tolist()  # init user_embed   np.array---list
        self.user_embed_v2 = self.get_user_vector()
        self.reject_item_embed = []
        self.reject_feature_embed = []
        self.reject_item_embed_mean = [0]*64
        self.reject_feature_embed_mean = [0]*64
        self.conver_his = [0] * self.max_turn  # conversation_history
        self.cand_len = [self.feature_length >>d & 1 for d in range(self.cand_len_size)][::-1]  #  Binary representation of candidate set length
        self.attr_ent = [0] * self.attr_state_num  #  attribute entropy
        self.one_hot=[0] * 29

        # init user prefer feature
        self._updata_reachable_feature(start='user')  # self.reachable_feature = []
        for i in (set(self.user_acc_feature) | set(self.user_rej_feature)):
            self.get_onehot(i)#attribute seq
        self.reachable_feature = list(set(self.reachable_feature) - set(self.user_acc_feature))
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1

        print('=== init user prefer large_feature: {}'.format(self.cur_node_set))
        self._update_cand_items(acc_feature=self.cur_node_set, rej_feature=[])
        self._update_feature_entropy()  # update entropy
        print('reset_reachable_feature num: {}'.format(len(self.reachable_feature)))

        #Sort reachable features according to the entropy of features
        reach_fea_score = self._feature_score()
        max_ind_list = []
        for k in range(self.ask_num):
            max_score = max(reach_fea_score)
            max_ind = reach_fea_score.index(max_score)
            reach_fea_score[max_ind] = 0
            max_ind_list.append(max_ind)
        max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
        [self.reachable_feature.pop(v - i) for i, v in enumerate(max_ind_list)]
        [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]

        return self._get_state()

    def _get_state(self):
        if self.command == 1:
            state = [self.user_embed, self.conver_his, self.attr_ent, self.cand_len]
            state = list(_flatten(state))
        elif self.command == 2: #attr_ent
            state = self.attr_ent
            state = list(_flatten(state))
        elif self.command == 3: #conver_his
            state = self.conver_his
            state = list(_flatten(state))
        elif self.command == 4: #cand_len
            state = self.cand_len
            state = list(_flatten(state))
        elif self.command == 5:  #user_embedding
            state = [self.conver_his, self.cand_len, self.user_embed_v2]#user_embedding is same to CRIF
            state = list(_flatten(state))
        elif self.command == 6: #attr_ent + conver_his + cand_len
            state = [self.conver_his, self.attr_ent, self.cand_len]
            state = list(_flatten(state))
        elif self.command == 7: #conver_his + cand_len
            state = [self.conver_his, self.cand_len]
            state = list(_flatten(state))
        # elif self.command == 8:
        #     state = [self.conver_his, self.attr_ent, self.cand_len]
        #     state = list(_flatten(state))
        elif self.command == 8: #conver_his + cand_len + attribute_seq(same to crm)
            state = [self.conver_his, self.cand_len, self.one_hot]
            state = list(_flatten(state))
        elif self.command == 9: #conver_his + cand_len + attribute_seq(same to crm)
            state = [self.conver_his, self.cand_len, self.user_embed]
            state = list(_flatten(state))
        elif self.command == 10: #conver_his + cand_len + reject_attribute_embed + reject_attribute_embed
            #state = [self.conver_his, self.cand_len, self.reject_feature_embed, self.reject_item_embed_mean]
            state = [self.conver_his, self.cand_len, self.reject_feature_embed_mean,self.reject_item_embed_mean]
            state = list(_flatten(state))
        return state

    def step(self, action):   #action:0  ask   action:1  recommend   setp=MAX_TURN  done
        done = 0
        recall_1,recall_5,recall_10=0.0,0.0,0.0
        MRR_1,MRR_5,MRR_10=0.0,0.0,0.0
        print('---------------step:{}-------------'.format(self.cur_conver_step))

        if self.cur_conver_step == self.max_turn:
            reward = self.reward_dict['until_T']
            self.conver_his[self.cur_conver_step-1] = self.history_dict['until_T']
            print('--> Maximum number of turns reached !')
            done = 1
        elif action == 0:   #ask large_feature
            print('-->action: ask features')
            reward, done, acc_feature, rej_feature = self._ask_update()  #update user's profile:  user_acc_feature & user_rej_feature & cand_items
            self._update_cand_items(acc_feature, rej_feature)  # update cand_item

            if len(acc_feature):   # can reach new large_featudate current node and reachable_feature
                self.cur_node_set = acc_feature
                self._updata_reachable_feature(start='large_feature')  # update user's profile: reachable_feature
                #compute feature_score:  fm_score  or fm_score+ent_score

            self.reachable_feature = list(set(self.reachable_feature) - set(self.user_acc_feature))
            self.reachable_feature = list(set(self.reachable_feature) - set(self.user_rej_feature))

            if self.command in [1, 2, 6, 7]:  # update attr_ent
                self._update_feature_entropy()
            if len(self.reachable_feature) != 0:  # if reachable_feature == 0 :cand_item= 1
                reach_fea_score = self._feature_score()  #  compute feature score
                max_ind_list = []
                for k in range(self.ask_num):
                    max_score = max(reach_fea_score)
                    max_ind = reach_fea_score.index(max_score)
                    reach_fea_score[max_ind] = 0
                    max_ind_list.append(max_ind)
                max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
                if self.command in [8]:
                    for i in (set(self.user_acc_feature) | set(self.user_rej_feature)):
                        self.get_onehot(i)#attribute seq
                [self.reachable_feature.pop(v - i) for i, v in enumerate(max_ind_list)]
                [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]


        elif action == 1:  #recommend items
            #select topk candidate items to recommend
            cand_item_score = self._item_score()
            item_score_tuple = list(zip(self.cand_items, cand_item_score))
            sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)
            self.cand_items, cand_item_score = zip(*sort_tuple)
            #===================== rec update=========
            reward, done, self.target_item_position= self._recommend_updata()
            if len(self.user_rej_item) > 0:
                for item_id in self.user_rej_item:
                    self.reject_item_embed.append(self.ui_embeds[self.user_length + item_id]) 
                self.reject_item_embed_mean=list(np.array(self.reject_item_embed).mean(axis=0))       
            #========================================
            #if reward == 1 or 0.99:
            if reward in [1,0.99]:
                print('-->Recommend successfully!')
                if  self.target_item_position > -1 and  self.target_item_position < 1:
                    recall_1=1.0
                    MRR_1=1/(self.target_item_position+1)
                    recall_5=1.0
                    MRR_5=1/(self.target_item_position+1)
                    recall_10=1.0
                    MRR_10=1/(self.target_item_position+1)
                elif  self.target_item_position > -1 and  self.target_item_position < 5:
                    recall_5=1.0
                    MRR_5=1/(self.target_item_position+1)
                    recall_10=1.0
                    MRR_10=1/(self.target_item_position+1)
                elif  self.target_item_position > -1 and  self.target_item_position < 10:
                    recall_10=1.0
                    MRR_10=1/(self.target_item_position+1)
                else:
                    print('rec fail')
            else:
                if self.command in [1, 2, 6, 7]:  # update attr_ent
                    self._update_feature_entropy()
                print('-->Recommend fail !')
        self.cur_conver_step += 1
        return self._get_state(), reward, done, recall_1,recall_5,recall_10,MRR_1,MRR_5,MRR_10

    def _updata_reachable_feature(self, start='large_feature'):
        self.reachable_feature = []
        if start == 'user':
            user_like_random_fea = random.choice(self.kg.G['item'][self.target_item]['belong_to_large'])
            self.user_acc_feature.append(user_like_random_fea)  # update user acc_fea
            self.cur_node_set = [user_like_random_fea]

            next_reachable_feature = []
            for cur_node in self.cur_node_set:
                fea_belong_items = list(self.kg.G['large_feature'][cur_node]['belong_to_large'])  # A-I

                cand_fea_belong_items = list(set(fea_belong_items) & set(self.cand_items))
                print('---> A-I-A item length: {}'.format(len(cand_fea_belong_items)))
                for item_id in cand_fea_belong_items:  # A-I-A   I in [cand_items]
                    next_reachable_feature.append(list(self.kg.G['item'][item_id]['belong_to_large']))
                next_reachable_feature = list(set(_flatten(next_reachable_feature)))
            self.reachable_feature = next_reachable_feature  # init reachable_feature

        elif start == 'large_feature':
            next_reachable_feature = []
            for cur_node in self.cur_node_set:
                fea_belong_items = list(self.kg.G['large_feature'][cur_node]['belong_to_large']) # A-I

                cand_fea_belong_items = list(set(fea_belong_items) & set(self.cand_items))
                for item_id in cand_fea_belong_items:  # A-I-A   I in [cand_items]
                    next_reachable_feature.append(list(self.kg.G['item'][item_id]['belong_to_large']))
                next_reachable_feature = list(set(_flatten(next_reachable_feature)))
            self.reachable_feature = next_reachable_feature

    def get_onehot(self,index_):
        self.one_hot[index_]=1

    def _feature_score(self):
        reach_fea_score = []
        for feature_id in self.reachable_feature:
            score = self.attr_ent[feature_id]
            reach_fea_score.append(score)
        return reach_fea_score

    def _item_score(self):
        cand_item_score = []
        for item_id in self.cand_items:
            item_embed = self.ui_embeds[self.user_length + item_id]
            score = 0
            score += np.inner(np.array(self.user_embed), item_embed)
            prefer_embed = self.feature_emb[self.acc_samll_fea, :]  #np.array (x*64), samll_feature
            for i in range(len(self.acc_samll_fea)):
                score += np.inner(prefer_embed[i], item_embed)
            cand_item_score.append(score)
        return cand_item_score

    def _ask_update(self):
        '''
        :return: reward, acc_feature, rej_feature
        '''
        done = 0

        feature_groundtrue = self.kg.G['item'][self.target_item]['belong_to_large']

        remove_acced_reachable_fea = self.reachable_feature.copy()   # copy reachable_feature

        acc_feature = list(set(remove_acced_reachable_fea[:self.ask_num]) & set(feature_groundtrue))
        rej_feature = list(set(remove_acced_reachable_fea[:self.ask_num]) - set(acc_feature))

        # update user_acc_feature & user_rej_feature
        self.user_acc_feature.append(acc_feature)
        self.user_acc_feature = list(set(_flatten(self.user_acc_feature)))
        self.user_rej_feature.append(rej_feature)
        self.user_rej_feature = list(set(_flatten(self.user_rej_feature)))

        if len(self.user_rej_feature) > 0 :
            for feature_id in self.user_rej_feature:
                self.reject_feature_embed.append(self.feature_emb[feature_id])
            self.reject_feature_embed_mean=list(np.array(self.reject_feature_embed).mean(axis=0))


        reward = None
        if len(acc_feature):
            if self.reward_pre==0:
                reward = self.reward_dict['ask_suc']
            elif self.reward_pre==1:
                reward = self.reward_dict_add_reward_pre['ask_suc']
            else:
                print('reward error')
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']   #update conver_his
        else:
            if self.reward_pre==0:
                reward = self.reward_dict['ask_fail']
            elif self.reward_pre==1:
                reward = self.reward_dict_add_reward_pre['ask_fail']
            else:
                print('reward error')
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']  #update conver_his

        if self.cand_items == []:  #candidate item set is empty
            done = 1
            reward = self.reward_dict['cand_none']
        return reward, done, acc_feature, rej_feature

    def _update_cand_items(self, acc_feature, rej_feature):

        small_feature_groundtrue = self.kg.G['item'][self.target_item]['belong_to']  # TODO small_ground truth
        if len(acc_feature):    #accept large_feature
            for feature_id in acc_feature:
                feature_small_ids = self.kg.G['large_feature'][feature_id]['link_to_feature']
                for small_id in feature_small_ids:
                    if small_id in small_feature_groundtrue:  # user_accept samll_tag
                        self.acc_samll_fea.append(small_id)
                        feature_items = self.kg.G['feature'][small_id]['belong_to']
                        self.cand_items = set(self.cand_items) & set(feature_items)   #  itersection
                    else:  #uesr reject small_tag
                        self.rej_samll_fea.append(small_id)  #reject no update

            self.cand_items = list(self.cand_items)

        self.cand_len = [len(self.cand_items) >>d & 1 for d in range(self.cand_len_size)][::-1]  # binary


    def _recommend_updata(self):
        print('-->action: recommend items')
        recom_items = self.cand_items[: self.rec_num]    # TOP k item to recommend
        reward=None
        if self.target_item in recom_items:
            if self.reward_pre == 0:
                reward = self.reward_dict['rec_suc']
            elif self.reward_pre == 1:
                reward = self.reward_dict_add_reward_pre['rec_suc']
            else:
                print('reward error')
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_scu'] #update state vector: conver_his
            done = 1
            self.target_item_position=recom_items.index(self.target_item)
        else:
            recom_items=list(recom_items)
            #print("recom_items",recom_items)
            self.user_rej_item.append(recom_items)
            self.user_rej_item=list(set(list(_flatten(self.user_rej_item))))
            if self.reward_pre == 0:
                reward = self.reward_dict['rec_fail']
            elif self.reward_pre == 1:
                reward = self.reward_dict_add_reward_pre['rec_fail']
                print('reward_fail',reward)
            else:
                print('reward error')
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_fail']  #update state vector: conver_his
            self.cand_items = self.cand_items[self.rec_num:]  #update candidate items
            self.cand_len = [len(self.cand_items) >> d & 1 for d in range(self.cand_len_size)][::-1]  #  binary
            done = 0
        return reward, done,self.target_item_position

    def _update_feature_entropy(self):
        if self.ent_way == 'entropy':
            cand_items_fea_list = []
            #TODO Dataframe
            for item_id in self.cand_items:
                cand_items_fea_list.append(list(self.kg.G['item'][item_id]['belong_to']))
            cand_items_fea_list = list(_flatten(cand_items_fea_list))
            self.attr_count_dict = dict(Counter(cand_items_fea_list))

            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent


            real_ask_able_large_fea = self.reachable_feature
            for large_fea_id in real_ask_able_large_fea:
                large_ent = 0
                small_feature = list(self.kg.G['large_feature'][large_fea_id]['link_to_feature'])
                small_feature_in_cand = list(set(small_feature) & set(self.attr_count_dict.keys()))

                for fea_id in small_feature_in_cand:
                    p1 = float(self.attr_count_dict[fea_id]) / len(self.cand_items)
                    p2 = 1.0 - p1
                    if p1 == 1:
                        large_ent += 0
                    else:
                        ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                        large_ent += ent
                self.attr_ent[large_fea_id] =large_ent
        elif self.ent_way == 'weight entropy':
            cand_items_fea_list = []
            self.attr_count_dict = {}
            cand_item_score = self._item_score()
            cand_item_score_sig = self.sigmoid(cand_item_score)  # sigmoid(score)

            for score_ind, item_id in enumerate(self.cand_items):
                cand_items_fea_list = list(self.kg.G['item'][item_id]['belong_to'])
                for fea_id in cand_items_fea_list:
                    if self.attr_count_dict.get(fea_id) == None:
                        self.attr_count_dict[fea_id] = 0
                    self.attr_count_dict[fea_id] += cand_item_score_sig[score_ind]

            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able_large_fea = self.reachable_feature
            sum_score_sig = sum(cand_item_score_sig)
            for large_fea_id in real_ask_able_large_fea:
                large_ent = 0
                small_feature = list(self.kg.G['large_feature'][large_fea_id]['link_to_feature'])
                small_feature_in_cand = list(set(small_feature) & set(self.attr_count_dict.keys()))

                for fea_id in small_feature_in_cand:
                    p1 = float(self.attr_count_dict[fea_id]) / sum_score_sig
                    p2 = 1.0 - p1
                    if p1 == 1 or p1 <= 0:
                        large_ent += 0
                    else:
                        ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                        large_ent += ent
                self.attr_ent[large_fea_id] = large_ent
    def sigmoid(self, x_list):
        x_np = np.array(x_list)
        s = 1 / (1 + np.exp(-x_np))
        return s.tolist()

    def get_user_vector(self):
        if self.data_name in [YELP]:
            highest_len = len(str(27675))
        div = pow(10,highest_len)
        result = float(self.user_id) / div
        user_vector = [result] * 4
        return user_vector

 


