import tensorflow as tf
import numpy
import pandas as pd
try:
    data = pd.read_csv('')
except Excetion as e:
    print ("An error occered {}".format(e))
    
x=data.iloc[:, :-1].values
y=data.iloc[:, -1].values

#Getting the mean, standard deviation, median, mode etc from the dataset by defining the `describe` to the pd dataframe
train_stats=data.describe()
train_stats.pop('output')

train_stas=train_stats.transpose()

y=data.pop('output')

def norm(x):
    return (x-train_stas['mean'])/train_stas['std']

norm_x=norm(data)

#shuffling the dataset with the buffer size (a.k.a random_points) to the length of the x we have

train_dataset=tf.data.Dataset.from_tensor_slices((x, y))
train_dataset=train_dataset.shuffle(buffer_size=len(x)).batch(32)

print (train_dataset)


inputs = tf.keras.Input(shape=(len(train_stats.columns))) 

#### INITIAL MODEL NEURONS ###
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
### END ###

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

main_loss=tf.keras.losses.BinaryCrossentropy()

main_optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)

main_metrics=tf.keras.metrics.BinaryAccuracy()

@tf.function()
def gradient_calculator (model, prediction_value, ground_truth_val):
    with tf.GradientTape() as tape:
        #Calculating the loss from the true values to the predicted values
        loss=main_loss(y_true=ground_truth_val, y_pred=model(prediction_value))

    # Calculating the gardient from the loss, trainable_variables
    grads=tape.gradient(loss, model.trainable_variables)
    
    main_optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train_for_epoch ():
    loss_dats=[]
    accuracy_data=[
    for (X_train , Y_train) in train_dataset:
        losses= gradient_calculator(model, X_train, Y_train)
        preds=model(X_train)
        
        
        loss_dats.append(losses)
        tf.print("Loss : {} accuracy: {}".format(losses,
                                                    main_metrics.result()))
        main_metrics.update_state(Y_train, preds)
        accuracy_data.append(main_metrics.result())
    return loss_dats, accuracy_data

epochs=10

for i in range (epochs):
    losses, acc=train_for_epoch()
    print ("    Epoch: {} \n Loss: {} \n Accuracy: {}".format(i, losses[len(losses)-1], acc[len(acc)-1]))

