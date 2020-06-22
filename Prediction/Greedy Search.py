def greedySearch(photo, model):
    in_text = 'sos'
    for i in range(max_length):
        sequence = [word_to_ix[w] for w in in_text.split() if w in word_to_ix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ix_to_word[yhat]
        in_text += ' ' + word
        if word == 'eos':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


