from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import reader

data, size, color_dict = reader.read_raw_data()
out_size = len(color_dict)
num_epochs = 500
lenx, leny = data.shape
num_classes = size + 1
total_series_length = lenx * (leny - 1)
truncated_backprop_length = leny - 1
state_size = 1
echo_step = 1
batch_size = 5
#batch_size = 3
num_layers = 4
num_batches = total_series_length//batch_size//truncated_backprop_length

def one_hot(num):
    arr = np.zeros(out_size)
    arr[num] = 1
    return arr

def generateData():
    x = data[:, 0 : leny -1]
    y = data[:, leny - 1 : leny]
    x = x.reshape((lenx * (leny - 1), 1))
    y = y.reshape((lenx, 1))

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))
    #print(y.shape)
    return (x, y)
print(truncated_backprop_length)
print("")
print("")
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, 1])
print(batchY_placeholder)
init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])
state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)]
)

W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)
W2 = tf.Variable(np.random.rand(state_size, out_size),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,out_size)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1)
#print(out_size)
#labels_series = batchY_placeholder
labels_series = tf.unstack(batchY_placeholder, axis=1)
print(">>>>>>>>>>>>>>>>>>>>")
print(len(labels_series))
labels = labels_series[0]
# Forward pass
cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
print(">>>>>>>>>>>>>>>>>>>>")
print(inputs_series)
print(rnn_tuple_state)
states_series, current_state = tf.nn.dynamic_rnn(cell, tf.expand_dims(batchX_placeholder, -1), initial_state=rnn_tuple_state)
states_series = tf.reshape(states_series, [batch_size, -1, state_size])
#logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
print(states_series.get_shape())
last = tf.gather(states_series, int(states_series.get_shape()[2]) - 1, axis=1)
print(">>>>>>>>>>>>>>>>>>>")
print(last)
#W2 = tf.Variable(np.random.rand(state_size, out_size),dtype=tf.float32)
#b2 = tf.Variable(np.zeros((1,out_size)), dtype=tf.float32)
logits = tf.matmul(last, W2) + b2
print(logits)
predictions_series = tf.nn.softmax(logits)
onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=out_size)
print(onehot_labels)
losses = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=logits)
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.05).minimize(total_loss)
print(train_step)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []
    x,y = generateData()
    for epoch_idx in range(num_epochs):
        _current_state = np.zeros((num_layers, 2, batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length
            start_idy = batch_idx * 1
            end_idy = start_idy + 1

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idy:end_idy]

            _total_loss, _train_step, _current_state, _predictions_series, _logits, _last = sess.run(
            [total_loss, train_step, current_state, predictions_series, logits, last],
            feed_dict={
                batchX_placeholder: batchX,
                batchY_placeholder: batchY,
                init_state: _current_state
            })
            print(batchY)
            print(_train_step)
            print(np.argmax(_predictions_series, 1))
            print(_predictions_series)
            print(_last)

            loss_list.append(_total_loss)

            if batch_idx%1 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                #plot(loss_list, _predictions_series, batchX, batchY)
    save_path = saver.save(sess, "./savedmodel/model.ckpt")
    print("Model saved in path: %s" % save_path)

plt.ioff()
plt.show()