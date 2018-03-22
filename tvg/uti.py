import os
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh

# 构建adj
def construct_adj():
    file_name_string="txt_graph.txt"#每一行是一个节点和它的8个邻居
    # adj=np.zeros((8172,8172))
    adj=sp.csr_matrix((8172,8172),dtype=int)
    with open(file_name_string,'rb') as filedata:
        for (num,line) in enumerate(filedata):
            row=line.split(' '.encode(encoding='utf-8'))
            for _ in row:
                col=int(_)
                if(col !=num):
                    adj[num,col]=1#将当前节点和邻居节点对应的元素值设为1
    return adj

# def 构建特征矩阵，列是词，行是doc
def load_data():
    filename='all_freq.txt'#是每一个词对应的频数
    all_words=[]
    with open(filename,'rb') as filedata:
        for line in filedata:
            word=line.split('\t'.encode(encoding='utf-8'))[0]
            word=str(word).split('\'')[-2]
            all_words.append(word)#将所有的词存为list，等会用来查词所在的列
        # print(all_words)
    # features=np.zeros((2360,8172),dtype=float)
    features=sp.lil_matrix((2360,8172))
    id_row=0
    dict_doc_to_id={'pos_brain_0003':0}#用来记录doc的索引，即行
    for info in os.listdir('txt_freq'):
        dict_doc_to_id[str(info[:-4])]=id_row
        domain=os.path.abspath('txt_freq')
        info=os.path.join(domain,info)
        with open(info,'rb') as words:
            for one_line in words:
                # print(one_line)
                w,val=one_line.strip().split('\t'.encode(encoding='utf-8'))
                w=str(w).split('\'')[-2]#每一行的词
                # print(w)
                val=str(val).split('\'')[-2]#词对应的频数
                val=int(val)
                if(w in all_words):#有可能txt_freq中的词，是all_freq没有的，只选有的词
                    id_col=all_words.index(w)
                    features[id_row,id_col]=val#在特征矩阵中，将id_row对应的文章中有的词，词的索引是id_col，该处的值设为频数
        id_row=id_row+1#用来计数第几篇文章，也是该文章在features的所在行

# print(adj[0][1655])
# print('dict_doc_to_id[pos_butterfly_0025]')
# ret=dict_doc_to_id['pos_cactus_0568']
# print(ret)

    # def construct_label():
    #将train，test，val数据集生成对应的标签矩阵和mask
    def get_docRow_cat(three_filename,y_array,mask_array):
        with open(three_filename,'rb') as trainset:
            for doc_cat in trainset:
                doc,cat=doc_cat.strip().split('\t'.encode(encoding='utf-8'))
                doc=str(doc).split('\'')[-2]
                cat=int(cat)
                doc_row=dict_doc_to_id[doc]
                y_array[doc_row][cat] = 1
                mask_array[doc_row]=True
                # print(doc_row)

    y_train=np.zeros((8172,10))
    y_test=np.zeros((8172,10))
    y_val=np.zeros((8172,10))
    train_mask=~np.ones((8172,),dtype=bool)
    test_mask=~np.ones((8172,),dtype=bool)
    val_mask=~np.ones((8172,),dtype=bool)
    # print(train_mask)
    get_docRow_cat('train_set.txt',y_train,train_mask)
    get_docRow_cat('test_set.txt',y_test,test_mask)
    get_docRow_cat('val_set.txt',y_val,val_mask)
    # print(train_mask[ret])

    adj=construct_adj()

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
# print('adj:',adj.shape)
# print(adj)
# print('features:',features.shape)
# print(features)
# print('y_train:',y_train.shape)
# print(y_train)
# print('y_test:',y_test.shape)
# print(y_test)
# print('y_val:',y_val.shape)
# print(y_val)
# print('train_mask:',train_mask.shape)
# print(train_mask)
# print('test_mask:',test_mask.shape)
# print(test_mask)
# print('val_mask:',val_mask.shape)
# print(val_mask)



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        # print('coords:', coords)
        # print('values:', values)
        # print('shape:', shape)
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # print("features:")
    # print(features)
    rowsum = np.array(features.sum(1))
    # print(rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)#dot函数是做矩阵乘法
    return sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)#稀疏矩阵
    rowsum = np.array(adj.sum(1))#将输入数据转换为ndarray
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.#判断是否无穷
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # print('adj--def preprpcess_adj')
    # print(adj)
    # print('adj.shape[0]:\n', adj.shape[0])
    # print('sp.eye():\n', sp.eye(adj.shape[0]))# 生成对角线为1的矩阵
    # print('adj+:', adj + sp.eye(adj.shape[0]))
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))#numpy.eye 生成对角矩阵
    # print('adj_nomalized:', adj_normalized)
    return sparse_to_tuple(adj_normalized)

def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)







