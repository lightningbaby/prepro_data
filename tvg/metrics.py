import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    # print("preds-def masked_accuracy in metrics.py-",preds,"\nlabels--",labels,"\nmask--",mask)
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))#对比这两个矩阵或者向量的相等的元素
    accuracy_all = tf.cast(correct_prediction, tf.float32)#将correct_prediction的数据格式转化成float32
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)#求平均值
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
