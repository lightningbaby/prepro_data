import os
import numpy as np

# 构建adj
def construct_adj():
    file_name_string="txt_graph.txt"#每一行是一个节点和它的8个邻居
    adj=np.zeros((8172,8172))
    with open(file_name_string,'rb') as filedata:
        for (num,line) in enumerate(filedata):
            row=line.split(' '.encode(encoding='utf-8'))
            for _ in row:
                col=int(_)
                if(col !=num):
                    adj[num][col]=1#将当前节点和邻居节点对应的元素值设为1
    return adj

# def 构建特征矩阵，行是词，列是doc
def load_data():
    filename='all_freq.txt'#是每一个词对应的频数
    all_words=[]
    with open(filename,'rb') as filedata:
        for line in filedata:
            word=line.split('\t'.encode(encoding='utf-8'))[0]
            word=str(word).split('\'')[-2]
            all_words.append(word)#将所有的词存为list，等会用来查词所在的列
        # print(all_words)
    features=np.zeros((2360,8172),dtype=int)
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
                    features[id_row][id_col]=val#在特征矩阵中，将id_row对应的文章中有的词，词的索引是id_col，该处的值设为频数
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













