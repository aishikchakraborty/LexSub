from typing import Dict
import json
import logging
import csv

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("lex_rel_classification")
class LexicalRelationClassificationDatasetReader(DatasetReader):
    """
    TODO(Kushal): Change this pydoc.
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    Expected format for each input line: {"paperAbstract": "text", "title": "text", "venue": "text"}

    The JSON could have other fields, too, but they are ignored.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        title: ``TextField``
        abstract: ``TextField``
        label: ``LabelField``

    where the ``label`` is derived from the venue of the paper.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for i, line in enumerate(data_file):
                if i == 0:
                    continue
                row = line.split()

                yield self.text_to_instance(word1=row[0], word2=row[1], pos=None, label=row[2])

    @overrides
    def text_to_instance(self, word1: str, word2: str, pos: str = None, label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        fields['word1'] = TextField(self._tokenizer.tokenize(word1), self._token_indexers)
        fields['word2'] = TextField(self._tokenizer.tokenize(word2), self._token_indexers)
        metadata = {'word1': word1, 'word2': word2}
        if label is not None:
            fields['label'] = LabelField(label)
            metadata['label'] = label
        if pos is not None:
            metadata['pos'] = pos

        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)


