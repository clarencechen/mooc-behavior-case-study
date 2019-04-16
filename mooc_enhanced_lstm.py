def enhanced_lstm_model(model_params,
        lstm_dropout: float = 0.2,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-6,
        confidence_penalty_weight: float = 0.1):

	word_ids = Input(shape=(model_params['seq_len'],), dtype='int32', name='word_ids')
    l2_regularizer = regularizers.l2(l2_reg_penalty)
    embedding_layer = ReusableEmbedding(
        model_params['e_vocab_size'],
        model_params['embed_dim'],
        input_length=model_params['seq_len'],
        name='bpe_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)
    output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='word_prediction_logits')

    output_softmax_layer = Softmax(name='word_predictions')

    next_step_input, embedding_matrix = embedding_layer(word_ids)

    for i in range(layers):
        next_step_input = LSTM(model_params['hidden_size'], dropout=lstm_dropout, return_sequences=True, name='LSTM '+str(i))(next_step_input)

    word_predictions = output_softmax_layer(
        output_layer([next_step_input, embedding_matrix]))
    model = Model(inputs=[word_ids], outputs=[word_predictions])
    # Penalty for confidence of the output distribution, as described in
    # "Regularizing Neural Networks by Penalizing Confident
    # Output Distributions" (https://arxiv.org/abs/1701.06548)
    confidence_penalty = K.mean(
        confidence_penalty_weight *
        K.sum(word_predictions * K.log(word_predictions), axis=-1))
    model.add_loss(confidence_penalty)

    return model