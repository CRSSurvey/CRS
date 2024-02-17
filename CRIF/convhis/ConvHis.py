import json
import random
from math import log, e, pow
from collections import defaultdict
import numpy as np
import torch
from utils import user_att_count_not_subset
from sklearn.metrics import roc_auc_score
from utils.global_variable import *
import torch.optim as optim # TODO: reject item update


def entropy(prob):
    if abs(prob)<1e-6 or abs(1-prob)<1e-6:
        return 0
    return - prob * log(prob, e) - (1-prob) * log(1-prob, e)


class ConvHis():
    def __init__(self, config):
        self.config = config.config
        self.user_num = config.user_num
        self.item_num = config.item_num
        self.attribute_num = config.attribute_num
        self.parent_attribute_num = config.parent_attribute_num
        self.att_pos_state = float(config.att_pos_state)
        self.att_neg_state = float(config.att_neg_state)
        self.item_neg_state = float(config.item_neg_state)
        self.init_state = float(config.init_state)
        self.max_conv_length = config.max_conv_length

        self.attribute_tree = config.att_tree_dict
        new_attribute_tree = {}
        for parent in self.attribute_tree:
            new_attribute_tree[int(parent)] = set(self.attribute_tree[parent])
        self.attribute_tree = new_attribute_tree

        self.attribute_parent_mat = np.zeros([self.parent_attribute_num, self.attribute_num])
        for attribute_parent, attribute_list in self.attribute_tree.items():
            for attribute in attribute_list:
                self.attribute_parent_mat[attribute_parent, attribute] = 1.

        self.user_info = config.user_info
        self.item_info = config.item_info

        self.user = None
        self.target_item = None
        self.candidate_list = None
        self.pos_attribute = None
        self.neg_attribute = None
        self.target_attribute = None
        self.not_target_attribute = None
        self.user_pos_item_list = None
        self.conv_neg_item_list = None
        self.convhis_vector = None
        self.conv_lenth = None
        self.asked_list = None

        self.rec = None
        self.attribute_entropy = None
        self.candidate_list_score = defaultdict(int)

        self.candidate_len = None
        self.target_rank = None


    def init_conv(self, user, target_item, init_pos_attribute_set, init_neg_attribute_set, init_parent_attribute):
        self.user = user
        self.target_item = target_item
        self.candidate_list = []
        self.current_agent_action = None
        self.attribute_entropy = None

        all_item_set = set([_ for _ in range(self.item_num)])
        candidate_set = set()
        # 选 candidate item set
        # contain all pos att & no neg att
        for i in all_item_set:
            if len(init_pos_attribute_set - self.item_info[i]) == 0 and \
                    len(init_neg_attribute_set & self.item_info[i]) == 0: # contain all pos att & no neg att
                candidate_set.add(i)
        self.candidate_list = list(candidate_set)

        all_att_set = set([_ for _ in range(self.attribute_num)])
        target_attribute_set = self.item_info[self.target_item]
        self.not_target_attribute = list(all_att_set - target_attribute_set)

        self.pos_attribute = init_pos_attribute_set # attribute value
        self.neg_attribute = init_neg_attribute_set
        self.target_attribute = list(self.item_info[target_item])
        self.user_pos_item_list = list(self.user_info[user])
        self.conv_neg_item_list = set()
        self.convhis_list = [self.init_state] * self.max_conv_length # [0] * 15
        self.convhis_list[0] = self.att_pos_state # 初始attribute 一定是对的
        self.conv_lenth = 1
        self.asked_list = [init_parent_attribute] # attribute type

        # TODO: reject item update
        self.item_optimizer = optim.Adam([param for param in self.rec.rec.parameters() if param.requires_grad == True], \
                                lr=self.config.item_lr, weight_decay=self.config.weight_decay)
        self.att_att_optimizer = optim.Adam([param for param in self.rec.rec.parameters() if param.requires_grad == True], \
                                lr=self.config.att_lr, weight_decay=self.config.weight_decay)
        self.item_att_optimizer = optim.Adam([param for param in self.rec.rec.parameters() if param.requires_grad == True], \
                                        lr=self.config.att_lr, weight_decay=self.config.weight_decay)
        # TODO: end

    def add_new_attribute(self, pos_attribute_set, parent_attribute):
        self.pos_attribute = self.pos_attribute.union(pos_attribute_set)
        self.neg_attribute = self.neg_attribute.union(self.attribute_tree[parent_attribute] - pos_attribute_set)

    def add_neg_attribute(self, neg_attribute_set):
        self.neg_attribute = self.neg_attribute.union(neg_attribute_set)

    def add_pos_attribute(self, pos_attribute_set):
        self.pos_attribute = self.pos_attribute.union(pos_attribute_set)

    def update_conv_his(self, pos, parent_attribute):
        if self.conv_lenth == self.max_conv_length:
            return
        if pos:
            self.convhis_list[self.conv_lenth] = self.att_pos_state
        else:
            self.convhis_list[self.conv_lenth] = self.att_neg_state
        self.conv_lenth += 1
        self.asked_list.append(parent_attribute)

    def add_conv_neg_item_list(self, neg_item_list):
        if self.conv_lenth == self.max_conv_length:
            return
        for item in neg_item_list:
            self.conv_neg_item_list.add(item)

        neg_item_set = set(neg_item_list)
        new_candidate_list = set(self.candidate_list) - neg_item_set
        self.candidate_list = list(new_candidate_list)

        self.convhis_list[self.conv_lenth] = self.item_neg_state
        self.conv_lenth += 1
        
    # 选candidate attribute的方式也是树的思路
    # initial pos attr: C, neg: G, all attr: A~G
    # candidate item: 有 pos attr(C) 无 neg attr(G), e.g., item1 (ABC), item2 (CDE), item3 (BC)
    # 常规思路candidate attr:  ABDEF,  根据树结构，initial pos attr——>item——>attr 选的candidate attr是 ABCDE
    def get_attribute_entropy(self):
        if self.attribute_entropy is None:
            attribute_count = defaultdict(int)
            for item in self.candidate_list:
                for att in self.item_info[item]: # candidate attr: 所有candidate item的attribute
                    attribute_count[att] += 1
            attribute_entropy_list = []
            for i in range(self.attribute_num):
                attribute_entropy_list.append(float(attribute_count[i])/len(self.candidate_list))
            attribute_entropy_list = np.array(list(map(entropy, attribute_entropy_list)))
            parent_attribute_entropy_list = np.matmul(self.attribute_parent_mat, attribute_entropy_list).tolist()
            for i in self.asked_list:
                parent_attribute_entropy_list[i] = 0.
            self.attribute_entropy = parent_attribute_entropy_list
        return self.attribute_entropy

    def get_max_attribute_entropy_index(self):
        entropy_list = self.get_attribute_entropy()
        max_score = max(entropy_list)
        max_score_index = entropy_list.index(max_score)
        return max_score_index

    def get_rank_attribute_entropy_index(self):
        entropy_list = self.get_attribute_entropy()
        indices = sorted(range(len(entropy_list)), key=lambda k: entropy_list[k], reverse=True)
        return indices

    def update_attribute_entropy(self):
        attribute_count = defaultdict(int)
        for item in self.candidate_list:
            if len(self.item_info[item] & self.neg_attribute) == 0:
                for att in self.item_info[item]:
                    attribute_count[att] += 1
        attribute_entropy_list = []
        for i in range(self.attribute_num):
            attribute_entropy_list.append(float(attribute_count[i])/len(self.candidate_list))
        attribute_entropy_list = np.array(list(map(entropy, attribute_entropy_list)))
        parent_attribute_entropy_list = np.matmul(self.attribute_parent_mat, attribute_entropy_list).tolist()
        for i in self.asked_list:
            parent_attribute_entropy_list[i] = 0.
        self.attribute_entropy = parent_attribute_entropy_list
        return self.attribute_entropy

    def get_available_items_for_recommend_feedback(self):
        candidate_att_item = user_att_count_not_subset.att_single_available_candidate_for_group(self.get_pos_attribute(), self.get_neg_attribute())
        return candidate_att_item

    def get_user_vertor(self):
        highest_len = len(str(self.user_num))
        div = pow(10, highest_len)
        result = float(self.user) / div
        user_vector = [result] * 4
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # print(self.user,highest_len,user_vector)
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
        return user_vector

    def get_recommend_feedback_length_vector(self):
        feedback_length = len(self.get_available_items_for_recommend_feedback())
        temp = '{:04b}'.format(feedback_length)
        feedback_length_vector = [int(temp[0])] + [int(temp[1])] + [int(temp[2])] + [int(temp[3])]
        return feedback_length_vector

    def get_convhis_vector(self):
        return self.convhis_list

    def get_length_vector(self):
        length_vector = [0.] * 8
        if len(self.candidate_list) <= 10:
            length_vector[0] = 1.
        if len(self.candidate_list) > 10 and len(self.candidate_list) <= 50:
            length_vector[1] = 1.
        if len(self.candidate_list) > 50 and len(self.candidate_list) <= 100:
            length_vector[2] = 1.
        if len(self.candidate_list) > 100 and len(self.candidate_list) <= 200:
            length_vector[3] = 1.
        if len(self.candidate_list) > 200 and len(self.candidate_list) <= 300:
            length_vector[4] = 1.
        if len(self.candidate_list) > 300 and len(self.candidate_list) <= 500:
            length_vector[5] = 1.
        if len(self.candidate_list) > 500 and len(self.candidate_list) <= 1000:
            length_vector[6] = 1.
        if len(self.candidate_list) > 1000:
            length_vector[7] = 1.
        return length_vector

    def get_user(self):
        return self.user

    def get_target_item(self):
        return self.target_item

    def get_pos_attribute(self):
        return list(self.pos_attribute)

    def get_neg_attribute(self):
        return list(self.neg_attribute)

    def get_target_attribute(self):
        return self.item_info[self.target_item]

    def get_user_pos_item_list(self):
        return self.user_info[self.user]

    def get_conv_neg_item_list(self):
        return list(self.conv_neg_item_list)

    def get_conv_length(self):
        return self.conv_lenth

    def get_candidate_list_len(self):
        return len(self.candidate_list)

    def get_candidate_list(self):
        return self.candidate_list

    def set_candidate_list(self, rank_list):
        self.candidate_list = rank_list

    def get_asked_list(self):
        return self.asked_list

    def set_rec(self, rec):
        self.rec = rec

    def get_item_score(self, item):
        user = self.get_user()
        pos_attribute = self.get_pos_attribute()
        neg_attribute = self.get_neg_attribute()

        item_score = self.rec.get_item_preference(user, pos_attribute, neg_attribute, item)
        item_score_list = item_score.cpu().numpy().tolist()
        min_score = min(item_score_list)
        margin_zero_score_list = [(x - min_score) for x in item_score_list]
        return margin_zero_score_list

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(x))
        return s

    def get_candidate_item_auc(self):
        item_label = []
        pos_number = 0
        pos_item_list = []
        for item in self.candidate_list:
            if self.item_info_conform_user(item):
                item_label.append(1)
                pos_number = pos_number + 1
                pos_item_list.append(item)
            else:
                item_label.append(0)
        item_score_list = self.rec.get_item_preference(self.user, self.pos_attribute, self.neg_attribute,
                                                       self.candidate_list)

        if len(set(item_label)) == 2:
            auc = roc_auc_score(item_label, item_score_list)
            print('positive_negative_item_auc', str(auc))
            print(pos_number)
            return auc
        else:
            print(pos_number)

    def get_target_item_auc(self):
        candidate_list = self.candidate_list.copy()
        candidate_list.remove(self.target_item)
        test_item = [self.target_item] + candidate_list
        test_item_label = [1] + [0] * len(candidate_list)
        item_score_list = self.rec.get_item_preference(self.user, self.pos_attribute, self.neg_attribute, test_item)

        auc = roc_auc_score(test_item_label, item_score_list)
        print('target_item_auc', str(auc))
        return auc

    def item_info_conform_user(self, item_id):
        # if self.pos_attribute.issubset(get_item_att(item_id)) and (
        #         len(self.neg_attribute & get_item_att(item_id)) == 0):
        #     return True
        # else:
            return False

    def get_candidate_len_and_target_rank(self, need_update=True):
        if need_update:
            candidate_list = self.candidate_list.copy()
            target_index = candidate_list.index(self.target_item)
            candidate_len = len(candidate_list)
            item_score_list = self.rec.get_item_preference(self.user, self.pos_attribute, self.neg_attribute,
                                                           candidate_list)
            values, indices = item_score_list.sort(descending=True)
            rank = indices.numpy().tolist().index(target_index)
            self.candidate_len = candidate_len
            self.target_rank = rank
            return self.candidate_len, self.target_rank
        else:
            return self.candidate_len, self.target_rank

    def get_candidate_len_and_target_rank_base_list(self, score_item_list):
        target_index = self.candidate_list.index(self.target_item)
        candidate_len = len(self.candidate_list)
        values, indices = score_item_list.sort(descending=True)
        rank = indices.numpy().tolist().index(target_index)
        self.candidate_len = candidate_len
        self.target_rank = rank
        return self.candidate_len, self.target_rank

    def set_candidate_len_and_target_rank(self, candidate_len, target_rank):
        self.candidate_len = candidate_len
        self.target_rank = target_rank

    def set_candidate_len_and_target_rank_and_can_list(self, candidate_len, target_rank, can_list):
        self.candidate_len = candidate_len
        self.target_rank = target_rank
        self.candidate_list = can_list

    def get_candidate_len_and_target_rank_for_ask(self, pos_att, neg_att):
        current_pos_att = self.pos_attribute
        current_neg_att = self.neg_attribute
        if len(pos_att) != 0: # 问对了
            new_candidate_list = list(self.get_pos_set_item(pos_att) & set(self.candidate_list))
            current_pos_att = current_pos_att.union(pos_att)
        else:
            new_candidate_list = self.candidate_list.copy()
            current_neg_att = current_neg_att.union(neg_att)

        candidate_len = len(new_candidate_list)
        target_index = new_candidate_list.index(self.target_item)

        item_score_list = self.rec.get_item_preference(self.user, current_pos_att, current_neg_att, new_candidate_list)
        values, indices = item_score_list.sort(descending=True)
        rank = indices.numpy().tolist().index(target_index)
        return candidate_len, rank, new_candidate_list

    def get_candidate_len_and_target_rank_for_rec(self, neg_item_list):
        neg_item_set = set(neg_item_list)
        new_candidate_list = set(self.candidate_list) - neg_item_set
        new_candidate_list = list(new_candidate_list)

        candidate_len = len(new_candidate_list)
        target_index = new_candidate_list.index(self.target_item)

        item_score_list = self.rec.get_item_preference(self.user, self.pos_attribute, self.neg_attribute, new_candidate_list)
        values, indices = item_score_list.sort(descending=True)
        rank = indices.numpy().tolist().index(target_index)
        return candidate_len, rank, new_candidate_list

    def get_candidate_len_and_target_rank_for_feedback_rec(self, neg_item_list, pos_att, neg_att):
        current_pos_att = self.pos_attribute
        current_neg_att = self.neg_attribute
        if len(pos_att) != 0:
            new_candidate_set = self.get_pos_set_item(pos_att) & set(self.candidate_list)
            current_pos_att = current_pos_att.union(pos_att)
        else:
            new_candidate_set = set(self.candidate_list)
            current_neg_att = current_neg_att.union(neg_att)

        neg_item_set = set(neg_item_list)
        new_candidate_set = new_candidate_set - neg_item_set
        new_candidate_list = list(new_candidate_set)

        candidate_len = len(new_candidate_list)
        target_index = new_candidate_list.index(self.target_item)

        item_score_list = self.rec.get_item_preference(self.user, current_pos_att, current_neg_att, new_candidate_list)
        values, indices = item_score_list.sort(descending=True)
        rank = indices.numpy().tolist().index(target_index)
        return candidate_len, rank, new_candidate_list

    def get_pos_set_item(self, pos_set):
        can_item = set()
        for att in pos_set:
            if len(can_item) == 0:
                can_item = self.config.att_info[att]
            else:
                can_item = can_item & self.config.att_info[att]
        return can_item

    # TODO: reject item update 
    def mini_update_FM(self):
        # This function is used to update the pretrained FM model.
        self.rec.init_train()
        # print('-----------------start----------------')
        user_list = torch.tensor(self.user)
        pos_items = random.sample(list(set(self.config.user_info[self.user]) - set([self.target_item])), 10)
        neg_items = list(self.conv_neg_item_list)[-10:]

        # to remove all ground truth interacted items ...
        random_neg = list(set(self.config.item_info.keys()) - set(self.config.user_info[self.user]))

        pos_item_list = pos_items + random.sample(random_neg, len(pos_items))  # add some random negative samples to avoid overfitting
        neg_item_list = neg_items + random.sample(random_neg, len(neg_items))  # add some random negative samples to avoid overfitting
        pos_item_list = torch.tensor(pos_item_list)
        neg_item_list = torch.tensor(neg_item_list)
        item_neg_item_list1 = torch.tensor(list(self.conv_neg_item_list))
        item_pos_att = torch.tensor(list(self.pos_attribute))
        item_neg_att = torch.tensor(list(self.neg_attribute))
        # print(user_list, pos_item_list,neg_item_list, item_pos_att, item_neg_att)
        neg_item_mask = torch.tensor([True] * len(neg_item_list))
        item_pos_att_mask = torch.tensor([True] * len(item_pos_att))
        item_neg_att_mask = torch.tensor([True] * len(item_neg_att))
        item_neg_item_list_mask1 = torch.tensor([True] * len(item_neg_item_list1))
        
        user_list = user_list.expand(len(neg_item_list)).cuda()
        item_pos_att_list = item_pos_att.expand(len(neg_item_list),-1).cuda()
        item_pos_att_mask = item_pos_att_mask.expand(len(neg_item_list),-1).cuda()
        item_neg_att_list = item_neg_att.expand(len(neg_item_list),-1).cuda()
        item_neg_att_mask = item_neg_att_mask.expand(len(neg_item_list),-1).cuda()
        item_neg_item_list = item_neg_item_list1.expand(len(neg_item_list),-1).cuda()
        item_neg_item_list_mask = item_neg_item_list_mask1.expand(len(neg_item_list),-1).cuda()

        pos_item_list = pos_item_list.cuda()
        neg_item_list1 = neg_item_list.unsqueeze(-1).cuda()
        neg_item_mask1 = neg_item_mask.unsqueeze(-1).cuda()

        # print(user_list.size(), item_pos_att_list.size(), item_pos_att_mask.size(),item_neg_att_list.size(), item_neg_att_mask.size(),
        #         pos_item_list.size(), neg_item_list1.size(), neg_item_mask1.size(),
        #         item_neg_item_list.size(), item_neg_item_list_mask.size())
        
        item_loss = self.rec.item_one_step_train_update(user_list, self.config.adj_index,
                                        item_pos_att_list, item_pos_att_mask,
                                        item_neg_att_list, item_neg_att_mask,
                                        pos_item_list, neg_item_list1, neg_item_mask1,
                                        item_neg_item_list, item_neg_item_list_mask)
        if item_loss == 0:
            return
        self.item_att_optimizer.zero_grad()
        item_loss.backward()
        self.item_att_optimizer.step()
        # print('-----------------out----------------')
    # TODO: end update

    # TODO: reject attribute update
    def mini_attr_update_FM(self, pos_attribute_set, parent_attribute):
        self.rec.init_train()
        # print('-----------------start----------------')
        user_list = torch.tensor(self.user)
        pos_att_list = list(set(self.config.item_info[self.target_item]) - set(self.pos_attribute))
        neg_att_list = torch.tensor(list(self.neg_attribute.union(self.attribute_tree[parent_attribute] - pos_attribute_set)))
        if len(pos_att_list) == 0:
            return
        if len(pos_att_list)>len(neg_att_list):
            pos_att_list = torch.tensor(pos_att_list[:len(neg_att_list)])
        else:
            pad = [pos_att_list[-1]]*(len(neg_att_list)-len(pos_att_list))
            pos_att_list = torch.tensor(pos_att_list+ pad)
        
        att_pos_att = torch.tensor(list(self.pos_attribute))
        att_neg_att = torch.tensor(list(self.neg_attribute))
        att_pos_att_mask = torch.tensor([True] * len(att_pos_att))
        att_neg_att_mask = torch.tensor([True] * len(att_neg_att))
        
        user_list = user_list.expand(len(neg_att_list)).cuda()
        att_pos_att_list = att_pos_att.expand(len(neg_att_list),-1).cuda()
        att_pos_att_mask = att_pos_att_mask.expand(len(neg_att_list),-1).cuda()
        att_neg_att_list = att_neg_att.expand(len(neg_att_list),-1).cuda()
        att_neg_att_mask = att_neg_att_mask.expand(len(neg_att_list),-1).cuda()
        att_pos_list = pos_att_list.cuda()
        att_neg_list = neg_att_list.cuda()
      

        # print(user_list.size(), att_pos_att_list.size(), att_pos_att_mask.size(),
        #       att_neg_att_list.size(), att_neg_att_mask.size(),
        #       att_pos_list.size(), att_neg_list.size())
        
        attr_loss = self.rec.att_one_step_train(user_list, self.config.adj_index,
                                            att_pos_att_list, att_pos_att_mask,
                                              att_neg_att_list, att_neg_att_mask,
                                              att_pos_list, att_neg_list)
        
        self.item_att_optimizer.zero_grad()
        attr_loss.backward()
        self.item_att_optimizer.step()
        # print(f'-----------------out----------------{attr_loss}')
    # TODO: end