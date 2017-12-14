import tensorflow as tf  
import numpy as np
from collections import Counter
import random
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

class SoftTreeNode:

    father = None
    left_child = None
    right_child = None
    is_leaf = 0
    levelID = -1;
    nodeID = -1;
    data_to_predict = np.array([])
    TargetDist = {} # target distribution of the leaf node
    w_tensor = None
    b_tensor = None
    sigma = None
    isLeftChild = -1
    isRightChild = -1
    alpha_tensor = None
    fai = None


    def __init__(self, sample_count, feature_count):
        pass
        

    # this is for training the soft decision tree
    def predict_proba(self, X):
        X = X[:, 0:(X.shape[1]-1)]
        abc = np.dot(X, self.w) + self.b
        return 1.0 / (1.0 + np.exp(-abc))

    # this is for prediction using the trained soft decision tree
    def predict_proba2(self, one_sample):
        abc = np.dot(one_sample, self.w) + self.b
        return 1.0 / (1.0 + np.exp(-abc))


class SoftTree:

    root = None
    height = -1
    all_levels = []
    all_levels_id = []

    def __init__(self, height, sample_count, feature_count):
        self.height = height
        node_counter = 0
        self.root = SoftTreeNode(sample_count, feature_count)
        self.root.father = None
        self.root.levelID = 0
        self.root.nodeID = node_counter
        node_counter = node_counter + 1

        cur_level = [self.root]
        cur_level_id = [self.root.nodeID]
        self.all_levels.append(cur_level)
        self.all_levels_id.append(cur_level_id)

        # internal nodes
        for i in range(1, height):
            tmp = []
            tmp_id = []
            for node in cur_level:
                lchild = SoftTreeNode(sample_count, feature_count)
                lchild.father = node
                lchild.levelID = i
                lchild.nodeID = node_counter
                node_counter = node_counter + 1
                lchild.isLeftChild= 1
                lchild.isRightChild = 0
                rchild = SoftTreeNode(sample_count, feature_count)
                rchild.father = node
                rchild.levelID = i
                rchild.nodeID = node_counter
                node_counter = node_counter + 1
                node.left_child = lchild
                node.right_child = rchild
                rchild.isLeftChild = 0
                rchild.isRightChild = 1

                if i == height-1:
                    lchild.is_leaf = 1
                    rchild.is_leaf = 1

                tmp.extend([lchild, rchild])
                tmp_id.extend([lchild.nodeID, rchild.nodeID])
            cur_level = tmp
            cur_level_id = tmp_id
            self.all_levels.append(cur_level)
            self.all_levels_id.append(cur_level_id)

    def clean(self):
        for level_id in range(0, self.height):
            cur_level = self.all_levels[level_id]    
            for node in cur_level:
                node.data_to_predict = np.array([])
                if level_id == self.height-1:
                    node.TargetDist = {}

    # this is for training the soft decision tree
    def predict(self, X):       
        self.root.data_to_predict = X

        # process internal nodes
        for level_id in range(0, self.height -1):
            cur_level = self.all_levels[level_id]  
            print('level id', level_id)        
            for node in cur_level:
                print("nodeid: ", node.nodeID)
                if node.data_to_predict.shape[0] > 0:
                    res_node = node.predict_proba(node.data_to_predict)
                    to_lchild = []
                    to_rchild = []
                    for i in range(0, len(res_node)):
                        rnum = random.random()
                        if rnum < res_node[i]:
                            to_rchild.append(node.data_to_predict[i, :])
                        else:
                            to_lchild.append(node.data_to_predict[i, :])

                    # make sure the shape
                    to_lchild = np.array(to_lchild)
                    to_rchild = np.array(to_rchild)
                    np.reshape(to_lchild, (-1, X.shape[1]))
                    np.reshape(to_rchild, (-1, X.shape[1]))

                    node.left_child.data_to_predict = to_lchild
                    node.right_child.data_to_predict = to_rchild

        # process leaf nodes
        leaf_level = self.all_levels[-1]
        for leaf in leaf_level:
            print('leaf id ', leaf.nodeID)
            if leaf.data_to_predict.shape[0] > 0:
                leaf.TargetDist = Counter(leaf.data_to_predict[:, -1])
                print('target distribution ')
                print(leaf.TargetDist)
                dlen = leaf.data_to_predict.shape[0]
                dlen = float(dlen)
                for key in leaf.TargetDist:
                    leaf.TargetDist[key] = leaf.TargetDist[key] / dlen
                print(leaf.TargetDist)


    # this is to predict using the trained soft decision tree
    def predict2(self, one_sample, feature_count): 
        cur_sample = one_sample[0:-1]      
        self.root.data_to_predict = cur_sample

        # process internal nodes
        cur_node = self.root      
        while cur_node.is_leaf != 1:
            res = cur_node.predict_proba2(cur_node.data_to_predict)
            rnum = random.random()
            if rnum < res[0]:
                to_rchild = cur_sample             
                cur_node.right_child.data_to_predict = to_rchild
                cur_node = cur_node.right_child
            else:
                to_lchild = cur_sample
                cur_node.left_child.data_to_predict = to_lchild
                cur_node = cur_node.left_child

        Q = np.exp(cur_node.fai) / np.sum(np.exp(cur_node.fai))
        return Q[1] # Q[1] is the probability of 1 and Q[0] is the probability of 0


def get_path_to_root(tree, node):
    res = []
    cur_node = node
    while cur_node is not None:
        res.append(cur_node)
        cur_node = cur_node.father

    return res


def train():
    # REMINDER: just support binary classification problem
    height = 4 # height of the soft decision tree

    # Credit_Card.csv is at https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset 
    df = pd.read_csv("Credit_Card.csv")
    print('full data shape')
    print(df.shape)
    df.rename(columns={'default.payment.next.month': 'default'}, inplace=True)
    positive = df[df['default']==1]
    print('positive shape')
    print(positive.shape)
    negative = df[df['default']==0]
    negative2 = negative.sample(n = 2*positive.shape[0], random_state = 0)
    print('negative2 shape')
    print(negative2.shape)
    df3 = positive.append(negative2, ignore_index=True)
    print('df3 shape')
    print(df3.shape)


    X = df3.values
    y = X[:, -1]
    X_train_before_scaler = X[:, 0:(X.shape[1]-1)] 
    scaler = StandardScaler()
    scaler.fit(X_train_before_scaler)
    X_train_after_scaler = scaler.transform(X_train_before_scaler)
    X_train_after_scaler = X_train_after_scaler.astype(np.float32)

    df2 = pd.DataFrame(X_train_after_scaler)
    df2['label'] = y

    df2_for_train = df2.sample(frac=0.8, random_state=1)
    df2_for_test = df2.loc[~df2.index.isin(df2_for_train.index)]
    print('df2 for train shape', df2_for_train.shape)
    print('df2 for test shape', df2_for_test.shape)


    X_train_for_tree = df2_for_train.values
    X_train_for_tensor = X_train_for_tree[:, 0:(X_train_for_tree.shape[1]-1)]
    X_train_for_tree = X_train_for_tree.astype(np.float32)
    X_train_for_tensor = X_train_for_tensor.astype(np.float32)
    print('X_train_for_tree shape', X_train_for_tree.shape)
    print('X_train_for_tensor shape', X_train_for_tensor.shape)
    
    sample_count = X_train_for_tensor.shape[0]
    feature_count = X_train_for_tensor.shape[1]
    tree = SoftTree(height, sample_count, feature_count)
    
    
    x = tf.placeholder(tf.float32, shape=(None, feature_count), name="x-input")

    leaf_level = tree.all_levels[-1]
    T_list = []
    fai_list = []
    Q_list = []
    ce_list = []
    for leaf in leaf_level:
        its_name = "T_leaf"+str(leaf.nodeID)
        T = tf.placeholder(tf.float32, shape=(2, 1), name=its_name) # just support binary classification problem
        T_list.append(T)

        fai = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
        fai_list.append(fai)

        Q = tf.exp(fai) / tf.reduce_sum(tf.exp(fai))
        Q_list.append(Q)

        ce = -tf.reduce_sum(T * tf.log(Q+1e-10))
        ce_list.append(ce)


    for level_id in range(0, height-1):
        cur_level = tree.all_levels[level_id]
        for node in cur_level:
            w = tf.Variable(tf.random_normal([feature_count, 1], stddev=1, seed=1))
            b = tf.Variable(tf.random_normal([1, 1], stddev=1, seed=1))
            node.w_tensor = w
            node.w_tensor = tf.clip_by_value(node.w_tensor, 1e-10, 1000.0)
            node.b_tensor = b
            node.sigma = 1.0 / (1.0 + tf.exp(-(tf.matmul(x, node.w_tensor)+node.b_tensor)))


    s_init = tf.constant(0.0, dtype=tf.float32)
    s = tf.Variable(np.zeros((sample_count, 1), np.float32))
    path_prob_list_init = []
    path_prob_list = []
    for i in range(0, len(leaf_level)):
        path_prob = tf.constant(1.0, shape=(sample_count, 1),  dtype=tf.float32)
        path_prob_list_init.append(path_prob)
        path_prob2 = tf.Variable(np.ones((sample_count, 1), np.float32))
        path_prob_list.append(path_prob2)
    
    for i in range(0, len(leaf_level)):
        leaf = leaf_level[i]
        path = get_path_to_root(tree, leaf)
        cur_node = leaf
        lev = 0
        for _ in range(0, len(path)-1):  
            father = cur_node.father          
            if cur_node.isLeftChild == 1:
                if lev == 0:
                    path_prob_list[i] = tf.multiply(path_prob_list_init[i], (1.0 - father.sigma))
                else:
                    path_prob_list[i] = tf.multiply(path_prob_list[i], (1.0 - father.sigma))
            if cur_node.isRightChild == 1:
                if lev == 0:
                    path_prob_list[i] = tf.multiply(path_prob_list_init[i], father.sigma)  
                else:
                    path_prob_list[i] = tf.multiply(path_prob_list[i], father.sigma)        
            cur_node = cur_node.father
            lev = lev + 1
        if i == 0:
            s = s_init + path_prob_list[i] * ce_list[i]
        else:
            s = s + path_prob_list[i] * ce_list[i]

    alpha_init = tf.constant(1.0, dtype=tf.float32)
    tree.root.alpha_tensor = tf.reduce_sum(tree.root.sigma) / sample_count
    for level_id in range(1, height-1):
        cur_level = tree.all_levels[level_id]
        for cur_internal_node in cur_level:
            path = get_path_to_root(tree, cur_internal_node)
            cur_node = cur_internal_node
            tmp_alpha = tf.Variable(np.zeros((sample_count, 1), np.float32))
            for j in range(0, len(path)-1):
                father = cur_node.father
                if cur_node.isLeftChild == 1:
                    if j == 0:
                        tmp_alpha = alpha_init * (1.0 - father.sigma)
                    else:
                        tmp_alpha = tmp_alpha * (1.0 - father.sigma)
                if cur_node.isRightChild == 1:
                    if j == 0:
                        tmp_alpha = alpha_init * father.sigma
                    else:
                        tmp_alpha = tmp_alpha * father.sigma
                cur_node = father
            denominator = tf.reduce_sum(tmp_alpha)
            numerator = tf.reduce_sum(tmp_alpha * cur_node.sigma)
            cur_internal_node.alpha_tensor = numerator / denominator

    C_init = tf.constant(0.0, dtype=tf.float32)
    C = tf.Variable(0.0, np.float32)
    count = 0
    for level_id in range(0, height-1):
        cur_level = tree.all_levels[level_id]
        for cur_internal_node in cur_level:
            a = 0.5*tf.log(cur_internal_node.alpha_tensor+1e-10) + 0.5*tf.log(1.0-cur_internal_node.alpha_tensor+1e-10)
            if count == 0:
                C = C_init + a
            else:
                C = C + a
            count = count + 1
    lam = 0.1
    C = -lam * C

    loss_op = tf.reduce_sum(tf.log(s)) 
    loss_op2 = (loss_op + C) / sample_count

    train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_op2) 

    loss2_list = []
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        for level_id in range(0, height-1):
            cur_level = tree.all_levels[level_id]
            for node in cur_level:
                node.w = node.w_tensor.eval()
                node.b = node.b_tensor.eval()

        tree.predict(X_train_for_tree)
        STEPS = 100
        for stepID in range(STEPS):
            print("Step -------------- ", stepID)  
            leaf_level = tree.all_levels[-1]
            myT_list = []
            for leaf in leaf_level:
                dict_of_leaf = dict(leaf.TargetDist) 
                e0 = 0.0
                e1 = 0.0
                if 0 in dict_of_leaf.keys():
                    e0 = dict_of_leaf[0]
                if 1 in dict_of_leaf.keys():
                    e1 = dict_of_leaf[1]
                myT_of_leaf = [e0, e1]
                myT_of_leaf = np.array(myT_of_leaf)
                myT_of_leaf = myT_of_leaf.astype(np.float32)
                myT_of_leaf = myT_of_leaf.reshape(2, 1)
                myT_list.append(myT_of_leaf)

            fdict = {}
            fdict[x] = X_train_for_tensor
            for i in range(0, len(myT_list)):
                fdict[T_list[i]] = myT_list[i]


            loss2_step, _ = sess.run([loss_op2, train_step], feed_dict=fdict)
            print("loss2_step: ", loss2_step)
            loss2_list.append(loss2_step)
            

            tree.clean()
            for level_id in range(0, height-1):
                cur_level = tree.all_levels[level_id]
                for node in cur_level:
                    node.w = node.w_tensor.eval()
                    node.b = node.b_tensor.eval()
            leaf_level = tree.all_levels[-1]
            for i in range(0, len(leaf_level)):
                leaf = leaf_level[i]
                leaf.fai = fai_list[i].eval()
            tree.predict(X_train_for_tree)
           
    print(loss2_list)

    # Now to do test
    X_test = df2_for_test.values
    prob_list = []
    for i in range(0, X_test.shape[0]):
        prob = tree.predict2(X_test[i, :], X_test.shape[1]-1)
        prob_list.append(prob)
    #accuracy = accuracy_score(X_test[:, -1], y_pred)
    auc = roc_auc_score(X_test[:, -1], prob_list)
    print("auc", auc)



if __name__=="__main__":

    train()

  