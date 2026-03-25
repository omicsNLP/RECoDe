from typing import Dict, List, Optional
from pydantic import BaseModel


class RelationInfons(BaseModel):
    type: str
    annotator: str
    update_at: Optional[str] = None


class Location(BaseModel):
    offset: int
    length: int


class AnnotationInfons(BaseModel):
    type: str
    input_value: Optional[str] = None
    input_type: Optional[str] = None
    identifier: Optional[str] = None
    annotator: Optional[str] = None
    updated_at: Optional[str] = None


class Annotation(BaseModel):
    id: str
    infons: AnnotationInfons

    text: Optional[str] = None

    locations: List[Location]


class Node(BaseModel):
    refid: str
    role: str
    annotation: Optional[Annotation] = None


class Relation(BaseModel):
    id: str
    infons: RelationInfons
    nodes: List[Node]


class Sentence(BaseModel):
    offset: int
    text: str
    annotations: Optional[List[Annotation]] = None
    relations: Optional[List[Relation]] = None


class Passage(BaseModel):
    offset: int
    infons: Optional[Dict] = None
    text: str
    sentences: Optional[List[Sentence]] = None
    annotations: List[Annotation]
    relations: Optional[List[Relation]] = None


def documents_to_instance(documents):
    return Instance(
        source="",
        date="",
        key="",
        infons=Infons(
            pmcid="",
            pmid="",
            link="",
            journal="",
            pub_type="",
            license="",
        ),
        documents=documents,
    )


class Document(BaseModel):
    id: str
    infons: Optional[Dict] = None
    passages: List[Passage]
    relations: List[Optional[dict]]


class LongForm(BaseModel):
    text: str
    extraction_algorithms: Optional[List] = None
    score: Optional[float] = None


class Abbreviation(BaseModel):
    short_form: str
    long_forms: List[LongForm]


class Infons(BaseModel):
    pmcid: Optional[str]
    pmid: Optional[str] = None
    doi: Optional[str] = None
    link: Optional[str]
    journal: Optional[str]
    pub_type: Optional[str]
    year: Optional[str] = None
    license: Optional[str] = None
    # custom fields
    abbreviations: List[Abbreviation] = None
    hybrid_abbreviations: List[Abbreviation] = None
    potential_abbreviations: List[Abbreviation] = None


class Instance(BaseModel):
    source: str
    date: str
    key: str
    infons: Infons
    documents: List[Document]


class ProcessedRelation(BaseModel):
    type: str
    type_candidates: Optional[List[dict]] = []

    Node1_str: str
    Node1_type: str
    Node1_sentence_offset: int
    Node1_original_offset: int
    Node1_length: int

    Node2_str: str
    Node2_type: str
    Node2_sentence_offset: int
    Node2_original_offset: int
    Node2_length: int

    verb_str: Optional[str] = None
    verb_sentence_offset: Optional[int] = None
    verb_original_offset: Optional[int] = None
    verb_length: Optional[int] = None

    original_txt: str
    original_txt_offset: int
    original_txt_info: Optional[Dict] = None

    def get_relation_str(self):
        return f"{self.type}\t{self.Node1_str}\t{self.Node1_type}\t{self.Node1_sentence_offset}\t{self.Node2_str}\t{self.Node2_type}\t{self.Node2_sentence_offset}\t{self.verb_str}\t{self.verb_sentence_offset}\t{self.original_txt}"

    def __eq__(self, value: object) -> bool:
        # Check if the compared object is an instance of ProcessedRelation
        if not isinstance(value, ProcessedRelation):
            return False

        # Check if Node1 and Node2 attributes match (type, offset, length)
        nodes_match = (
            self.Node1_type == value.Node1_type
            and self.Node1_original_offset == value.Node1_original_offset
            and self.Node1_length == value.Node1_length
            and self.Node2_type == value.Node2_type
            and self.Node2_original_offset == value.Node2_original_offset
            and self.Node2_length == value.Node2_length
        )

        # Check if the original text offset matches
        txt_offset_match = self.original_txt_offset == value.original_txt_offset

        # Check if original text info matches specific fields
        txt_info_match = False
        if self.original_txt_info and value.original_txt_info:
            txt_info_match = (
                self.original_txt_info.get("pmc_id")
                == value.original_txt_info.get("pmc_id")
                and self.original_txt_info.get("doc_num")
                == value.original_txt_info.get("doc_num")
                and self.original_txt_info.get("passage_num")
                == value.original_txt_info.get("passage_num")
                and self.original_txt_info.get("sentence_num")
                == value.original_txt_info.get("sentence_num")
            )
        elif not self.original_txt_info and not value.original_txt_info:
            # If both are None, they match
            txt_info_match = True

        # Return True if all conditions match
        return nodes_match and txt_offset_match and txt_info_match

    def get_hash_key(self):
        return (
            f"{self.Node1_type}_{self.Node1_original_offset}_{self.Node1_length}_"
            + f"{self.Node2_type}_{self.Node2_original_offset}_{self.Node2_length}_"
            + f"{self.original_txt_offset}_"
            + f"{'_'.join([f'{k}:{v}' for k, v in sorted(self.original_txt_info.items())]) if self.original_txt_info else ''}"
        )

    def __hash__(self) -> int:
        # Create a string representation of the key attributes
        hash_string = self.get_hash_key()
        # Generate a hash from the string representation
        return hash(hash_string)
