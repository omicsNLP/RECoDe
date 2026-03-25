import json

from .type_def import *
from .sentence_splitter import sentence_split_spans


def parse_json_instance(path):
    with open(path, "r") as f:
        instance_dict = json.load(f)
    try:
        instance = Instance(**instance_dict)
    except Exception as e:
        raise ValueError(f"Error parsing JSON instance: {e}")

    # add sentences
    for document in instance.documents:
        for passage in document.passages:
            _sentences = []
            # sentence_str_list = sent_tokenize(passage.text)
            _sentence_spans = []
            _sentence_spans = sentence_split_spans(passage.text)

            # for sen_start_idx, sen_end_idx in sentence_tokenizer.span_tokenize(passage.text):
            for sen_start_idx, sen_end_idx in _sentence_spans:
                # split_sentences

                _sentence_str = passage.text[sen_start_idx:sen_end_idx]
                _sentence_offset = passage.offset + sen_start_idx
                _sentence = Sentence(offset=_sentence_offset, text=_sentence_str)
                _sentence_annotations = []

                for annotation in passage.annotations:
                    if len(annotation.locations) != 1:
                        raise ValueError(
                            f"Annotation {annotation.id} has multiple locations, which is not supported."
                        )
                    location = annotation.locations[0]
                    if (location.offset >= _sentence_offset) and (
                        location.offset < passage.offset + sen_end_idx
                    ):
                        _sentence_annotations.append(annotation)
                _sentence.annotations = _sentence_annotations

                _annotation_dict = {_anno.id: _anno for _anno in _sentence_annotations}

                _sentence_relations = []
                for relation in passage.relations:
                    _is_all_include_sentence = True
                    for _node in relation.nodes:
                        if _node.refid in _annotation_dict:
                            _node.annotation = _annotation_dict[_node.refid]
                        else:
                            _is_all_include_sentence = False
                    if _is_all_include_sentence:
                        _sentence_relations.append(relation)
                _sentence.relations = _sentence_relations
                _sentences.append(_sentence)
            passage.sentences = _sentences
    return instance
