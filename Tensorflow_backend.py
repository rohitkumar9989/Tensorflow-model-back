import tensorflow as tf
import numpy
import pandas as pd

data = pd.read_csv('C:\\Users\\rohit\\Desktop\\Python programmes\\All python stuff\\New folder (2)\\heart.csv')
x=data.iloc[:, :-1].values
y=data.iloc[:, -1].values

train_stats=data.describe()
train_stats.pop('output')
train_stas=train_stats.transpose()
y=data.pop('output')

def norm(x):
    return (x-train_stas['mean'])/train_stas['std']

norm_x=norm(data)

train_dataset=tf.data.Dataset.from_tensor_slices((x, y))
train_dataset=train_dataset.shuffle(buffer_size=len(x)).batch(32)
print (train_dataset)


inputs = tf.keras.Input(shape=(len(train_stats.columns))) #<-- please enter the number of classes you have
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

main_loss=tf.keras.losses.BinaryCrossentropy()
main_optimizer=tf.keras.optimizers.RMSprop()
main_metrics=tf.keras.metrics.BinaryAccuracy()

def gradient_calculator (model, prediction_value, ground_truth_val):
    with tf.GradientTape() as tape:
        loss=main_loss(y_true=ground_truth_val, y_pred=model(prediction_value))

    grads=tape.gradient(loss, model.trainable_variables)
    main_optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train_for_epoch ():
    loss_dats=[]
    for (X_train , Y_train) in train_dataset:
        losses= gradient_calculator(model, X_train, Y_train)
        preds=model(X_train)
        loss_dats.append(losses)
        tf.print("Loss : {} accuracy: {}".format(losses,
                                                    main_metrics.result()))
        main_metrics.update_state(Y_train, preds)
    return loss_dats

epochs=3

for i in range (epochs):
    losses=train_for_epoch()
    print ("Epoch {}".format(i))

