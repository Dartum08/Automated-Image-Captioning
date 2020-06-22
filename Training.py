samples_per_epoch = 0
for ca in all_train_captions:
    samples_per_epoch += len(ca.split())-1

samples_per_epoch # = 306404



val_samples_per_epoch = 0
for key, caps in val_desc.items():

  for ca in caps:

    val_samples_per_epoch += len(ca.split())

val_samples_per_epoch # = 46622


model.compile(loss='categorical_crossentropy', optimizer='adam')
number_pics_per_bath = 128
steps = samples_per_epoch//number_pics_per_bath
val_steps = val_samples_per_epoch//number_pics_per_bath


from keras.callbacks import ModelCheckpoint
checkpoint_filepath = '/content/drive/My Drive/Image Captioning Data/mc.ckpt'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_best_only=False)


# This function keeps the initial learning rate for the first ten epochs  
# and decreases it exponentially after that.  
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.10)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

round(model.optimizer.lr.numpy(), 5)


train_gen = data_generator(train_desc, train_features, word_to_ix, max_length, number_pics_per_bath)
val_gen = data_generator(val_desc, val_features, word_to_ix, max_length, number_pics_per_bath)


model.fit(train_gen, epochs=1, steps_per_epoch=steps, verbose=1, validation_data=val_gen, validation_steps=val_steps, 
                    callbacks=[model_checkpoint_callback])