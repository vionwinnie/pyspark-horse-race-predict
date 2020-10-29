import tensorflow as tf

## Set up a tensorflow sequential model
def create_model(lr,dropout):
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
