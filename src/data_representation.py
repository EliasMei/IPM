import os
import csv


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label: int = None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None, left_id=None, right_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.left_id = left_id
        self.right_id = right_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines



class BERTProcessor(DataProcessor):
    """Processor for preprocessed BERT data sets (abt_buy, company, etc.)"""

    def get_train_examples(self, data, with_text_b=True, oov_method=None, vocab=None):
        """See base class."""
        return self._create_examples(data, "train", with_label=True, with_text_b=with_text_b, oov_method=oov_method, vocab=vocab)
    
    def get_dev_examples(self, data, with_text_b=True, oov_method=None, vocab=None):
        """See base class."""
        return self._create_examples(data, "dev", with_label=True, with_text_b=with_text_b, oov_method=oov_method, vocab=vocab)
    
    def get_test_examples(self, data, with_text_b=True, oov_method=None, vocab=None):
        """See base class."""
        return self._create_examples(data, "test", with_label=False, with_text_b=with_text_b, oov_method=oov_method, vocab=vocab)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, data, set_type, with_label, with_text_b=True, oov_method=None, vocab=None):
        """Creates examples for the training and dev sets."""
        examples = []
        def transform(text, method, vocab):
            tokens = text.split()
            if method == "unk":
                tokens = [w if w in vocab else "<unk>" for w in tokens]
            elif method == "del":
                tokens = [w for w in tokens if w in vocab]
            return " ".join(tokens)

        for _, row in data.iterrows():
            guid = "%s-%s" % (set_type, row.id)
            try:
                text_a = row.text_a
                text_b = row.text_b if with_text_b else None
                if oov_method == 'del' and vocab is not None:
                    text_a = transform(text_a, 'del', vocab)
                    if with_text_b:
                        text_b = transform(text_b, 'del', vocab)
                if oov_method == 'unk' and vocab is not None:
                    text_a = transform(text_a, 'unk', vocab)
                    if with_text_b:
                        text_b = transform(text_b, 'unk', vocab)
                label = row.label if with_label else None
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
