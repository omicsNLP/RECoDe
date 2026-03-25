from datetime import datetime
import recode


class CoDietRelationCandidateGenerator:
    def __init__(self, target_tuple_types):
        self.target_tuple_types = target_tuple_types

        self.target_tuple_types_to_dict = dict()
        for _target_elem in self.target_tuple_types:
            _e1 = _target_elem[0]
            _e2 = _target_elem[1]

            if _e1 not in self.target_tuple_types_to_dict:
                second_target_set = set()
                self.target_tuple_types_to_dict[_e1] = second_target_set
            else:
                second_target_set = self.target_tuple_types_to_dict[_e1]
            second_target_set.add(_e2)
        self.current_time = datetime.now()
        self.current_time_str = self.current_time.strftime("%Y:%m:%d_%H:%M")

    def get_target_tuple_candidates(self, sentence):
        _candidate_e1_dict = {}

        for anno in sentence.annotations:
            _anno_type = anno.infons.type
            if _anno_type in self.target_tuple_types_to_dict:
                if _anno_type not in _candidate_e1_dict:
                    _e1_candidates = []
                    _candidate_e1_dict[_anno_type] = _e1_candidates
                else:
                    _e1_candidates = _candidate_e1_dict[_anno_type]
                _e1_candidates.append(anno)

        if len(_candidate_e1_dict) == 0:
            return []

        _target_pair_candidates = []

        for anno in sentence.annotations:
            _anno_type = anno.infons.type

            _anno_start = anno.locations[0].offset
            _anno_end = int(anno.locations[0].offset) + int(anno.locations[0].length)

            for _e1_type, _e1_candidates in _candidate_e1_dict.items():
                _e2_types = self.target_tuple_types_to_dict[_e1_type]
                if _anno_type in _e2_types:
                    for _e1_candidate in _e1_candidates:
                        _e1_start = _e1_candidate.locations[0].offset
                        _e1_end = int(_e1_candidate.locations[0].offset) + int(
                            _e1_candidate.locations[0].length
                        )

                        if not (_e1_end <= _anno_start or _anno_end <= _e1_start):
                            continue

                        _target_pair_candidates.append([_e1_candidate, anno])

        return _target_pair_candidates

    def convert_passage(self, text, sen_offset, entities):

        _print_str = ""
        _past_print_idx = 0

        _err = False
        _error_str = ""

        sorted_entities = sorted(entities, key=lambda x: x.locations[0].offset)

        for idx, e in enumerate(sorted_entities):
            offset = e.locations[0].offset
            length = e.locations[0].length
            e_type = e.infons.type
            e_text = e.text

            _start_idx = int(offset) - int(sen_offset)
            _end_idx = int(offset) + int(length) - int(sen_offset)

            _print_str += (
                f"{text[_past_print_idx:_start_idx]}<{e_type}>{e_text}</{e_type}>"
            )
            _past_print_idx = _end_idx

        _print_str += f"{text[_past_print_idx:]}"

        return _print_str

    def inference_with_sentence(self, sentence):

        _target_pair_candidates = self.get_target_tuple_candidates(sentence)

        _relation_cnt = 0
        relations = []
        if len(_target_pair_candidates) > 0:
            for e1, e2 in _target_pair_candidates:
                _question_input = {
                    "e1_type": e1.infons.type,
                    "e1_text": e1.text,
                    "e2_type": e2.infons.type,
                    "e2_text": e2.text,
                }
                _converted_sen = self.convert_passage(
                    sentence.text, sentence.offset, [e1, e2]
                )

                rel_infons = recode.RelationInfons(
                    type="CoDiet_dummy",
                    annotator=f"CoDiet_dummy",
                    update_at=self.current_time_str,
                )

                e1_node = recode.Node(refid=e1.id, role="e1")
                e2_node = recode.Node(refid=e2.id, role="e2")
                rel = recode.Relation(
                    id=f"{sentence.offset}_r_{_relation_cnt}",
                    infons=rel_infons,
                    nodes=[e1_node, e2_node],
                )
                relations.append(rel)
                _relation_cnt += 1

        sentence.relations = relations

    def inference_with_instance(self, instance):
        for document in instance.documents:
            for passage in document.passages:
                for sentence in passage.sentences:
                    self.inference_with_sentence(sentence)
