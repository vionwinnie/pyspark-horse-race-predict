import tensorflow as tf
import os,shutil

def set_callback():
    callback_dir = "/home/dcvionwinnie/output/callback/"
    weight_cur_epoch = callback_dir + "{epoch:02d}-weight-validation-loss-{val_loss:.4f}.hdf5"
    
    print("========================================================")
    print(os.path.abspath(callback_dir))
    if os.path.exists(callback_dir):
        shutil.rmtree(callback_dir)

    os.makedirs(callback_dir)
    weight_callback = tf.keras.callbacks.ModelCheckpoint(
        weight_cur_epoch,
        verbose=1,
        save_weights_only=True,
        period=5)
    return os.path.abspath(callback_dir),weight_callback

## Set up a tensorflow sequential model
def build(lr,dropout):
    """
    input:
    lr = learning rate
    dropout = percentage of dropout nodes[0-1]

    output:
    model : compiled tensorflow model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(96, activation='relu', input_shape=(104,)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(14, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  #metrics=[tf.keras.metrics.Precision(name='precision')]
                  metrics=['accuracy',tf.keras.metrics.Precision(name='precision')])

    return model

if __name__=="__main__":
    learn_rate = 1e-04
    epoch = 200
    batch_size = 32
    dropout=0.5

    model = create_model(learn_rate,dropout)
    print("model created")
