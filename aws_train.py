"""
Script to train pixel_cnn on AWS
"""
import os
import pickle
import tensorflow as tf
import numpy as np

from utils import pixelcnn_made, data_batch_iterator

if __name__ == "__main__":
    # create pixelCNN-MADE model
    features_ph, labels_ph, probs, loss, train_op = pixelcnn_made()

    # load data and split in train/tst
    np.random.seed(0)  # so that the split is always the same
    with (open("mnist-hw1.pkl", "rb")) as openfile:
        data = pickle.load(openfile)
    t_data = data["train"]
    test_data = data["test"]

    n_train = len(t_data)
    indices = np.random.choice(np.arange(0, n_train), n_train, replace=False)
    train_data = t_data[indices[:(n_train * 80) // 100]]
    val_data = t_data[indices[(n_train * 80) // 100:]]

    # ********** TRAINING **********
    batch_size = 128
    n_steps = 2000

    batch_iterator = data_batch_iterator(train_data, batch_size)
    steps = []
    train_losses = []
    val_losses = []

    save_path = os.path.join(os.getcwd(), "model/model.ckpt")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, save_path)

        for step in range(n_steps + 1):
            batch_data = next(batch_iterator)
            batch_features = batch_data
            batch_labels = batch_data
            sess.run(train_op, feed_dict={features_ph: batch_features,
                                          labels_ph: batch_labels})

            if step % 100 == 0:
                indices = np.random.choice(np.arange(0, len(val_data)), 32, replace=False)
                batch_val_data = val_data[indices]
                train_loss = sess.run(loss, feed_dict={features_ph: batch_features,
                                                       labels_ph: batch_labels})
                val_loss = sess.run(loss, feed_dict={features_ph: batch_val_data,
                                                     labels_ph: batch_val_data})

                steps.append(step)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                with open("losses.pkl", "bw") as file:
                    pickle.dump({"val_losses": val_losses,
                                 "steps": steps,
                                 "train_losses": train_losses}, file)


            if step % 100 == 0:
                saver.save(sess, save_path)
                indices = np.random.choice(np.arange(0, len(val_data)), 32, replace=False)
                batch_val_data = val_data[indices]
                train_loss = sess.run(loss, feed_dict={features_ph: batch_features,
                                                       labels_ph: batch_labels})
                val_loss = sess.run(loss, feed_dict={features_ph: batch_val_data,
                                                     labels_ph: batch_val_data})
                print("step {:>5} train|val avg_nll: {:>5}|{}".format(
                    step,
                    round(train_loss, 3),
                    round(val_loss, 3)
                ))
