from keras import regularizers
from keras.models import Model
# noinspection PyPep8Naming
from keras import backend as K
from keras.layers import Input, Softmax, Embedding, Add, Lambda, Dense

from multihot_utils import MultihotEmbedding
from embedding_utils import TiedOutputLayer

from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerACT, TransformerBlock


def universal_transformer_gpt_model(
        max_seq_length: int, vocabulary_size: int,
        word_embedding_size: int, transformer_depth: int,
        num_heads: int, transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-6,
        confidence_penalty_weight: float = 0.1):
    """
    A model which is similar to the one described by OpenAI in paper
    "Improving Language Understanding by Generative Pre-Training", except
    that it relies L2 regularization of the word embedding matrix
    (instead of the dropout), and uses Universal Transformer architecture.
    """
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty
                      else None)
    embedding_layer = Embedding(
        vocabulary_size, word_embedding_size,
        input_length=max_seq_length,
        name='token_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)
    output_layer = TiedOutputLayer(embedding_layer,
        activation='softmax',
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='word_predictions')
    coordinate_embedding_layer = TransformerCoordinateEmbedding(
        transformer_depth,
        name='coordinate_embedding')
    transformer_act_layer = TransformerACT(name='adaptive_computation_time')
    transformer_block = TransformerBlock(
        name='transformer', num_heads=num_heads,
        residual_dropout=transformer_dropout,
        attention_dropout=transformer_dropout,
        use_masking=True, vanilla_wiring=False)

    next_step_input, embedding_layer(word_ids)
    act_output = next_step_input

    for i in range(transformer_depth):
        next_step_input = coordinate_embedding_layer(next_step_input, step=i)
        next_step_input = transformer_block(next_step_input)
        next_step_input, act_output = transformer_act_layer(next_step_input)

    transformer_act_layer.finalize()
    next_step_input = act_output
    word_predictions = output_layer(next_step_input)
    model = Model(inputs=[word_ids], outputs=[word_predictions])
    # Penalty for confidence of the output distribution, as described in
    # "Regularizing Neural Networks by Penalizing Confident
    # Output Distributions" (https://arxiv.org/abs/1701.06548)
    confidence_penalty = K.mean(
        confidence_penalty_weight *
        K.sum(word_predictions * K.log(word_predictions), axis=-1))
    model.add_loss(confidence_penalty)
    return model


def vanilla_transformer_gpt_model(
        max_seq_length: int, vocabulary_size: int,
        word_embedding_size: int, transformer_depth: int,
        num_heads: int, transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-6,
        confidence_penalty_weight: float = 0.1,
        multihot_input = False):
    """
    A model which is almost identical to the one described by OpenAI in paper
    "Improving Language Understanding by Generative Pre-Training", except
    that it uses L2 regularization of the word embedding matrix,
    instead of the dropout.
    """
    if multihot_input:
        word_ids = Input(shape=(max_seq_length, vocabulary_size), dtype='float', name='multihot_ids')
    else:
        word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty
                      else None)
    if multihot_input:
        embedding_layer = MultihotEmbedding(
            vocabulary_size, word_embedding_size,
            input_length=max_seq_length,
            name='multihot_embeddings',
            # Regularization is based on paper "A Comparative Study on
            # Regularization Strategies for Embedding-based Neural Networks"
            # https://arxiv.org/pdf/1508.03721.pdf
            embeddings_regularizer=l2_regularizer)
    else:
        embedding_layer = Embedding(
            vocabulary_size, word_embedding_size,
            input_length=max_seq_length,
            name='token_embeddings',
            # Regularization is based on paper "A Comparative Study on
            # Regularization Strategies for Embedding-based Neural Networks"
            # https://arxiv.org/pdf/1508.03721.pdf
            embeddings_regularizer=l2_regularizer)
    output_layer = TiedOutputLayer(embedding_layer,
        activation='softmax',
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='word_predictions')
    coordinate_embedding_layer = TransformerCoordinateEmbedding(
        1,
        name='coordinate_embedding')

    next_step_input = embedding_layer(word_ids)
    next_step_input = coordinate_embedding_layer(next_step_input, step=0)
    for i in range(transformer_depth):
        next_step_input = (
            TransformerBlock(
                name='transformer' + str(i), num_heads=num_heads,
                residual_dropout=transformer_dropout,
                attention_dropout=transformer_dropout,
                vanilla_wiring=True)
            (next_step_input))

    word_predictions = output_layer(next_step_input)
    model = Model(inputs=[word_ids], outputs=[word_predictions])
    # Penalty for confidence of the output distribution, as described in
    # "Regularizing Neural Networks by Penalizing Confident
    # Output Distributions" (https://arxiv.org/abs/1701.06548)
    confidence_penalty = K.mean(
        confidence_penalty_weight *
        K.sum(word_predictions * K.log(word_predictions), axis=-1))
    model.add_loss(confidence_penalty)
    return model
