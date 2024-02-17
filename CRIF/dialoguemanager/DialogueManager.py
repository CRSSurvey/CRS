import torch

from utils.utils import would_user_further_response, set_random_seed

def trans_index(origin_dict):
    new_dict = {}
    for key in origin_dict:
        new_dict[int(key)] = origin_dict[key]
    return new_dict


class DialogueManager:
    def __init__(self, config, rec, agent, user, convhis):
        self.config = config
        self.attribute_tree = config.config.att_tree_dict
        self.attribute_tree = trans_index(self.attribute_tree)

        self.ask_action_index = config.ask_action_index #0
        self.rec_action_index = config.rec_action_index #2
        self.rec_with_att_feedback_action_index = config.rec_with_att_feedback_action_index # 1
        self.rec_success_reward = config.rec_success_reward #1
        self.pos_attribute_reward = config.pos_attribute_reward #0.1
        self.neg_attribute_reward = config.neg_attribute_reward #0.03
        self.neg_feedback_attribute_reward = config.neg_feedback_attribute_reward #0.03
        self.rec_with_att_feedback_action_index = config.rec_with_att_feedback_action_index # 1
        self.user_quit_reward = config.user_quit_reward #-0.3
        self.every_turn_reward = config.every_turn_reward # -0.1
        self.turn_limit = config.turn_limit #15

        self.rec = rec
        self.agent = agent
        self.user = user
        self.convhis = convhis

        self.target_user = None
        self.target_item = None
        self.silence = True
        self.turn_num = None
        self.current_agent_action = None
        self.current_neg_item_set = set()
        set_random_seed(1)

    def initialize_dialogue(self, target_user, target_item, silence, given_init_parent_attribute=None):
        self.target_user = target_user
        self.target_item = target_item
        self.silence = silence
        self.turn_num = 1
        self.current_agent_action = None

        self.agent.init_episode()
        init_pos_attribute_set, init_neg_attribute_set, init_parent_attribute \
            = self.user.init_episode(target_user, target_item, given_init_parent_attribute)
        self.convhis.init_conv(target_user, target_item, init_pos_attribute_set, init_neg_attribute_set,
                               init_parent_attribute)

        return list(init_pos_attribute_set)

    def user_turn(self, ask_attribute_list, ask_item_list):
        if ask_attribute_list != None:
            attribute_list = self.user.next_turn(ask_attribute_list)
            return attribute_list
        if ask_item_list != None:
            item_list = []
            for item in ask_item_list:
                if item == self.target_item:
                    item_list.append(item)
                    break
            return item_list

    def user_turn_for_rec_att_feedback(self, ask_attribute_list):
        if ask_attribute_list != None:
            attribute_list = self.user.next_turn(ask_attribute_list)
            return attribute_list

    def user_turn_for_rec_feedback(self, ask_item_list):
        target_item_info = self.config.config.item_info[self.target_item]
        accept_attribute = set()
        for item in ask_item_list:
            item_info = self.config.config.item_info[item]
            if item_info.issubset(target_item_info):
                accept_attribute = accept_attribute.union(item_info)
        return accept_attribute

    def get_current_agent_action(self):
        return self.current_agent_action

    def initialize_episode(self, target_user, target_item, given_init_parent_attribute=None, silence=True):
        self.target_user = target_user
        self.target_item = target_item
        self.turn_num = 1
        self.current_agent_action = None
        self.silence = silence
        
        # 获取初始parent attribute，和negative and positive attribute value set
        # 随机找一个 target item的target attribute里任一parent attribute作为init_parent_attribute
        # target item的target attribute value中属于init_parent_attribute的 作为 init_pos_attribute_set
        # 剩下的init_parent_attribute的所有attribute value作为init_neg_attribute_set
        init_pos_attribute_set, init_neg_attribute_set, init_parent_attribute \
            = self.user.init_episode(target_user, target_item, given_init_parent_attribute) 
        # conv_his, length, asked_attribute type list, positive_attribute value, 
        # negative_attribute value, candidate item，target_attribute 初始化
        self.convhis.init_conv(target_user, target_item, init_pos_attribute_set, init_neg_attribute_set, init_parent_attribute)

        return self.get_state()

    def step(self, action_index):
        is_ask = action_index == self.ask_action_index # 0 ask
        is_rec_feed = action_index == self.rec_with_att_feedback_action_index # 1

        candidate_list = self.convhis.get_candidate_list()

        conv_neg_item_list = set(self.convhis.get_conv_neg_item_list())
        # current_neg_item_set 是否全部包含 conv_neg_item_list, e.g., conv_neg_item_list {1,2,3,4,5}, current_neg_item_set {1,2,3,4}
        # conv_neg_item_list - self.current_neg_item_set = {5}, 就是上一轮对话新增的reject item
        neg_item_list = conv_neg_item_list - self.current_neg_item_set 
        if neg_item_list != None: # 否  更新current_neg_item_set
            self.current_neg_item_set = conv_neg_item_list
        ask_item_list, cand_score_list = self.rec.get_recommend_item_list(self.target_item, list(neg_item_list), candidate_list)
        action_index = self.convhis.get_max_attribute_entropy_index() # 获取entropy 最大的attribute type(parent attr) index（用item个数算）
        # feedback_rec_item_list，获取rec的item 列表， scoring item + attr—rule item
        # feedback_att_list：所有attr-rule选的item的所有 attribute
        feedback_rec_item_list, feedback_att_list = self.get_feedback_recommend(ask_item_list) 
        if len(self.convhis.get_pos_attribute()) == 2 and len(feedback_att_list) > 0:
            print("pos: ", self.convhis.get_pos_attribute(), " item: ", feedback_rec_item_list[-len(feedback_att_list):],  " att: ", feedback_att_list, " ; u: ", self.target_user, " ; i: ", self.target_item)

        # 把所有按照attribute rule-base 选出来的candidate item的attribute区分一下, 后面用于inference，e.g., item1：attr1, attr2, attr3
        # feedback_attribute_list = attr1, attr2
        # feedback_pos_attribute_set = attr1, attr2
        # feedback_neg_attribute_set = attr3
        feedback_attribute_list = self.user_turn_for_rec_att_feedback(feedback_att_list)
        feedback_pos_attribute_set = set(feedback_attribute_list)
        feedback_neg_attribute_set = set(feedback_att_list) - feedback_pos_attribute_set
        # print('_____________________________________________________________')
        # print(feedback_att_list, self.convhis.get_pos_attribute(), self.user.pos_attribute_set, feedback_pos_attribute_set)
        # print('_____________________________________________________________')

        be_len, be_rank = self.convhis.get_candidate_len_and_target_rank_base_list(cand_score_list) # candidate length, target item rank

        feedback_rec_len = 0
        feedback_rec_rank = 0
        feedback_rec_list = []
        feedback_rec_success = False
        feedback_item_list = self.user_turn(None, feedback_rec_item_list)
        if len(feedback_item_list) == 0: # 推错了; 更新candidate_item_len, target item rank, new_candidate_item_list
            feedback_rec_len, feedback_rec_rank, feedback_rec_list = self.convhis.get_candidate_len_and_target_rank_for_feedback_rec(feedback_rec_item_list, feedback_pos_attribute_set, feedback_neg_attribute_set)
        else: # 推对了
            feedback_rec_success = True

        ask_attribute_list = self.attribute_tree[action_index]
        attribute_list = self.user_turn(ask_attribute_list, None) # 问对，问错之后 pos attribute的更新
        pos_attribute_set = set(attribute_list)
        neg_attribute_set = set(ask_attribute_list) - pos_attribute_set
        # 更新candidate_item_len, target item rank, new_candidate_item_list
        ask_len, ask_rank, ask_list = self.convhis.get_candidate_len_and_target_rank_for_ask(pos_attribute_set,
                                                                                             neg_attribute_set)

        success = False
        step_state = -1

        if is_ask:
            self.convhis.set_candidate_len_and_target_rank_and_can_list(ask_len, ask_rank, ask_list)
            ask_reward = self.agent.get_reward(self.get_reward_state(self.ask_action_index)) - torch.tensor(0.1).cuda()
            feedback_rec_reward = self.agent.get_reward(self.get_reward_state(self.rec_with_att_feedback_action_index)) - torch.tensor(0.1).cuda()
            reward = ask_reward

            ask_attribute_list = self.attribute_tree[action_index]
            attribute_list = self.user_turn(ask_attribute_list, None)
            pos_attribute_set = set(attribute_list)
            neg_attribute_set = set(ask_attribute_list) - pos_attribute_set # 父attribute下面所有不是pos的都是neg

            if len(pos_attribute_set) == 0: # 问错了
                step_state = "0" + str(list(neg_attribute_set)[0])
            else: # 问对了
                step_state = "1" + str(list(pos_attribute_set)[0])

            self.convhis.add_new_attribute(pos_attribute_set, action_index)
            self.convhis.update_conv_his(len(pos_attribute_set) > 0, action_index)
            self.convhis.update_attribute_entropy()
            # TODO: reject attribute update
            # if len(list(neg_attribute_set))>0:
            #     self.convhis.mini_attr_update_FM(pos_attribute_set, action_index)
            # TODO: end 
            if len(attribute_list) > 0:
                IsOver = False
            else:
                IsOver = False
        else: # rec
            # if feedback_rec_success == True:
            #     print('---------------------------------------------')
            #     print(feedback_rec_len, feedback_rec_rank, feedback_rec_list)
            #     print('---------------------------------------------')
            self.convhis.set_candidate_len_and_target_rank_and_can_list(feedback_rec_len, feedback_rec_rank, feedback_rec_list)
            ask_reward = self.agent.get_reward(self.get_reward_state(self.ask_action_index)) - torch.tensor(0.1).cuda()
            feedback_rec_reward = self.agent.get_reward(self.get_reward_state(self.rec_with_att_feedback_action_index)) - torch.tensor(0.1).cuda()

            item_list = self.user_turn(None, feedback_rec_item_list)
            uf = would_user_further_response() # return false
            if uf:
                accept_att = self.user_turn_for_rec_att_feedback(feedback_rec_item_list)
                if len(accept_att) > 0:
                    self.convhis.add_pos_attribute(accept_att)

            if len(item_list) > 0: # 推对了
                IsOver = True
                success = True
                feedback_rec_reward = feedback_rec_reward + torch.tensor(1.0).cuda()

                step_state = "2"
            else: # 推错了
                if len(feedback_att_list) > 0: # 如果有按照attr-rule选择的item
                    step_state = "4" + str(feedback_rec_item_list[-1]) + "qwe" + str(feedback_att_list[-1])
                else: # 全部是按item score选的item，没有符合attribute-rule的item
                    step_state = "3" + str(feedback_rec_item_list[0])


                self.convhis.add_pos_attribute(feedback_pos_attribute_set) # TODO: reduce inference
                self.convhis.add_neg_attribute(feedback_neg_attribute_set) # TODO: reduce inference
                self.convhis.add_conv_neg_item_list(feedback_rec_item_list)
                # TODO: reject item update 
                self.convhis.mini_update_FM()
                # TODO: end update
                IsOver = False

            reward = feedback_rec_reward

        self.turn_num += 1
        if self.turn_num == self.turn_limit + 1: # 超过最大轮数未推成功 quit
            reward = reward - torch.tensor(0.3).cuda()
            if success:
                success = False
            if not IsOver: 
                IsOver = True
        # return IsOver, self.get_state(), reward, success, be_len, be_rank, ask_len, ask_rank, \
        #        feedback_rec_len, feedback_rec_rank, ask_reward, feedback_rec_reward, step_state
        # metric
        rec_result = None
        if success:
            rec_result = [feedback_rec_item_list, self.target_item]
            # print('____________________________________________________________')
            # print (feedback_rec_item_list, self.target_item, feedback_rec_rank)
            # print('____________________________________________________________')
        # if len(feedback_att_list) > 0:
        #     print(f'*************************{step_state}*************************')
        #     print(feedback_neg_attribute_set, (self.convhis.neg_attribute))
        #     print(f'**************************************************')
        return IsOver, self.get_state(), reward, success, be_len, be_rank, ask_len, ask_rank, \
               feedback_rec_len, feedback_rec_rank, ask_reward, feedback_rec_reward, step_state, rec_result 
        # metric end

    def get_state(self):
        state_user = self.convhis.get_user_vertor()
        state_convhis = self.convhis.get_convhis_vector().copy()
        state_len = self.convhis.get_length_vector().copy()
        dialogue_state = state_user + state_convhis + state_len
        dialogue_state = torch.tensor(dialogue_state)
        return dialogue_state

    def get_reward_state(self, action_index):
        state_user = self.convhis.get_user_vertor()
        state_convhis = self.convhis.get_convhis_vector().copy()
        state_len = self.convhis.get_length_vector().copy()
        dialogue_state = state_user + state_convhis + state_len + [action_index]
        dialogue_state = torch.tensor(dialogue_state)
        return dialogue_state

    def get_feedback_recommend(self, norm_rec_list):
        candidate_att_item = self.convhis.get_available_items_for_recommend_feedback()
        candidate_item_list = list(candidate_att_item.keys())
        candidate_att_list = list(candidate_att_item.values()) # attr based item : three rules
        feedback_num = len(candidate_att_item)
        if feedback_num > 5:
            feedback_num = 5
        rec_item_list = norm_rec_list[:(10 - feedback_num)]
        rec_item_list = rec_item_list + candidate_item_list[:feedback_num] # [rating based + attr based item]
        # rec_item_list = candidate_item_list[:feedback_num] + rec_item_list # why not use this one? 
        return rec_item_list, candidate_att_list[:feedback_num]