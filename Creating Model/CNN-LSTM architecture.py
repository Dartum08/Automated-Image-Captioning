def create_model():

  inputs1 = Input(shape=(2048,))
  fe1 = Dropout(0.8)(inputs1)
  fe3 = Dense(256, activation='relu')(fe1)
  inputs2 = Input(shape=(max_length,))
  se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
  se2 = Dropout(0.8)(se1)
  se4 = LSTM(256)(se2)
  decoder1 = add([fe3, se4])
  decoder2 = Dense(256, activation='relu')(decoder1)
  outputs = Dense(vocab_size, activation='softmax')(decoder2)
  model = Model(inputs=[inputs1, inputs2], outputs=outputs)

  # Setting embedding layer to non-trainable so that embedding matrix won't get updated
  model.layers[2].set_weights([embedding_matrix])
  model.layers[2].trainable = False

  return model


model = create_model()
#model.summary()