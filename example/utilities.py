import cPickle
import os
import theano

from fuel.datasets.billion import OneBillionWord
from fuel.transformers import Batch, Mapping, Filter
from fuel.transformers.sequences import NGrams
from fuel.schemes import ConstantScheme

floatX = theano.config.floatX


def reject_repeated_words(sentence):
    """
    Remove sentences that contain more than 5 times the same consecutive
    character.
    """
    c = 1
    prev_word = None
    for w in sentence[0]:
        if w == prev_word:
            c += 1
            if c > 5:
                return False
        else:
            c = 1
        prev_word = w
    return True


def build_stream(dataset, n_grams, batch_size, times=None):
    example_stream = dataset.get_example_stream()

    example_stream = Filter(example_stream, reject_repeated_words)
    n_gram_stream = NGrams(n_grams, example_stream)

    batch_stream = Batch(n_gram_stream,
                         ConstantScheme(batch_size, times=times),
                         strictness=1)

    def reshape(batch):
        return batch[0].astype("int32"), batch[1][:, None].astype("int32")

    return Mapping(batch_stream, reshape)


def create_streams(batch_size, n_grams):

    print 'Loading vocab...',
    vocab, _ = cPickle.load(
        open(os.path.join(os.environ['DATA_PATH'],
                          '1-billion-word', 'billion_vocab_793471.pkl'), 'r'))
    print 'Loaded!'

    # Create datasets and streams
    train = OneBillionWord('training', range(1, 100), vocab, bos_token=None)
    train_stream = build_stream(train, n_grams, batch_size)
    valid = OneBillionWord('heldout', range(1), vocab, bos_token=None)
    valid_stream = build_stream(valid, n_grams, batch_size)

    return train_stream, valid_stream, vocab
