from lxmls import DATA_PATH
import lxmls
import lxmls.sequences.crf_online as crfo
import lxmls.readers.pos_corpus as pcc
import lxmls.sequences.id_feature as idfc
import lxmls.sequences.extended_feature as exfc
from lxmls.readers import pos_corpus
from lxmls.sequences import structured_perceptron 


corpus = lxmls.readers.pos_corpus.PostagCorpus()

train_seq = corpus.read_sequence_list_conll(DATA_PATH + "/train-02-21.conll", 
                                            max_sent_len=10, max_nr_sent=1000)

test_seq = corpus.read_sequence_list_conll(DATA_PATH + "/test-23.conll", 
                                           max_sent_len=10, max_nr_sent=1000)

dev_seq = corpus.read_sequence_list_conll(DATA_PATH + "/dev-22.conll", 
                                          max_sent_len=10, max_nr_sent=1000)

## Build features 
feature_mapper = idfc.IDFeatures(train_seq)
feature_mapper.build_features()

## Train the StructuredPerceptron implemented previously
sp = structured_perceptron.StructuredPerceptron(corpus.word_dict, 
                                                corpus.tag_dict,
                                                feature_mapper)
sp.num_epochs = 20
sp.train_supervised(train_seq)


## Evaluate the StructuredPerceptron model

pred_train = sp.viterbi_decode_corpus(train_seq)
pred_dev = sp.viterbi_decode_corpus(dev_seq)
pred_test = sp.viterbi_decode_corpus(test_seq)

eval_train = sp.evaluate_corpus(train_seq, pred_train)
eval_dev = sp.evaluate_corpus(dev_seq, pred_dev)
eval_test = sp.evaluate_corpus(test_seq, pred_test)

print("Structured Perceptron - ID Features Accuracy Train: %.3f Dev: %.3f Test: %.3f"\
      %( eval_train,eval_dev,eval_test))

feature_mapper = exfc.ExtendedFeatures(train_seq)
feature_mapper.build_features()

sp = structured_perceptron.StructuredPerceptron(corpus.word_dict,
                                                corpus.tag_dict, 
                                                feature_mapper)
sp.num_epochs = 20
sp.train_supervised(train_seq)

## Evaluate the StructuredPerceptron model with the extended features

pred_train = sp.viterbi_decode_corpus(train_seq)
pred_dev = sp.viterbi_decode_corpus(dev_seq)
pred_test = sp.viterbi_decode_corpus(test_seq)

eval_train = sp.evaluate_corpus(train_seq, pred_train)
eval_dev = sp.evaluate_corpus(dev_seq, pred_dev)
eval_test = sp.evaluate_corpus(test_seq, pred_test)

print("SP_ext -  Accuracy Train: %.3f Dev: %.3f Test: %.3f"\
      %(eval_train,eval_dev, eval_test))
