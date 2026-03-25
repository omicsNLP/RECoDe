from ..classifier_base import ClassifierBase


class C4MiddleAsso(ClassifierBase):
    def __init__(self, **kwargs):
        current_labels = [
            "IncreaseAssociation",
            "DecreaseAssociation",
            "GeneralAssociation",
        ]
        current_label_to_label = {
            "(A) IncreaseAssociation": "increaseAssociation",
            "(B) DecreaseAssociation": "decreaseAssociation",
            "(C) GeneralAssociation": "GeneralAssociation",
        }
        super().__init__(current_labels, current_label_to_label, **kwargs)

    def prompt(self, entity1, entity2, sentence):
        system_prompt = (
            "You are a Clinical Scientific Reviewer specializing in biomedical relation extraction. "
            "Your task is to analyze given texts and extract meaningful relationships between biomedical entities. "
            "Provide accurate, concise, and well-structured explanations based on scientific principles."
        )

        user_prompt = (
            f"Determine the specific type of association between '{entity1}' and '{entity2}' in the following sentence: '{sentence}'. "
            f"\n\nAnswer only with one of the following options:\n"
            f"(A) IncreaseAssociation - '{entity1}' increases or enhances '{entity2}'.\n"
            f"(B) DecreaseAssociation - '{entity1}' decreases or inhibits '{entity2}'.\n"
            f"(C) GeneralAssociation - '{entity1}' and '{entity2}' are related, but the nature of the effect is unclear "
            f"(e.g., general association, causal relationship without specifying increase or decrease).\n\n"
            f"**Guidelines for classification:**\n"
            f"- (A) IncreaseAssociation: Choose this if '{entity1}' increases, enhances, or boosts '{entity2}'.\n"
            f"- Use (A) if '{entity1}' leads to or is linked with **higher** levels, prevalence, or activity of '{entity2}'.\n"
            f"- (B) DecreaseAssociation: Choose this if '{entity1}' decreases, inhibits, or suppresses '{entity2}'.\n"
            f"- These directional cues may be **inside or outside the entity span**, and may appear as verbs (e.g., 'increases', 'reduces') or adjectives/nouns (e.g., 'higher', 'lower', 'reduction in').\n"
            f"- Use (B) if '{entity1}' leads to or is linked with **lower** levels, prevalence, or activity of '{entity2}'.\n"
            f"- (C) GeneralAssociation: Choose this if '{entity1}' and '{entity2}' are related but the sentence does not explicitly state an increase or decrease effect.\n"
            f"Use (C) GeneralAssociation if the sentence describes a **causal relationship** (e.g., '{entity1}' causes '{entity2}') without specifying whether it increases or decreases.\n"
            f"- Use (C) only if there's no directional cue or causal wording.\n\n"
            f"**Examples:**\n"
            f"- (A) IncreaseAssociation: 'X supplementation leads to increased Y levels.'\n"
            f"- (A) IncreaseAssociation: 'Obesity is associated with higher hypertension rates.'\n"
            f"- (B) DecreaseAssociation: 'Folate supplementation leads to a reduction in homocysteine levels.'\n"
            f"- (B) DecreaseAssociation: 'X intake reduces Y levels significantly.'\n"
            f"- (C) GeneralAssociation: 'X and Y are linked in metabolic pathways.'\n\n"
            f"- (C) GeneralAssociation: 'X causes Y in a biological process.' (without indication of increase or decrease)\n"
            f"- (C) GeneralAssociation: 'X leads to Y without stating whether it increases or decreases.'\n\n"
            f"**Provide your answer in the following format:**\n"
            f"Reason: Explain the relationship focusing on key verbs and negations if any.\n"
            f"Answer: (A) IncreaseAssociation OR (B) DecreaseAssociation OR (C) GeneralAssociation\n"
        )

        return system_prompt, user_prompt
