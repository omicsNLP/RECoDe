from ..classifier_base import ClassifierBase


class C1First(ClassifierBase):
    def __init__(self, **kwargs):
        current_labels = ["Yes", "NoAssociation", "Unrelated"]
        current_label_to_label = {
            "(A) Yes": "TopAssociation",
            "(B) NoAssociation": "NoAssociation",
            "(C) Unrelated": "Unrelated",
        }
        super().__init__(current_labels, current_label_to_label, **kwargs)

    def prompt(self, entity1, entity2, sentence):
        system_prompt = (
            "You are a Clinical Scientific Reviewer specializing in biomedical relation extraction. "
            "Your task is to analyze given texts and extract meaningful relationships between biomedical entities. "
            "Provide accurate, concise, and well-structured explanations based on scientific principles."
        )

        user_prompt = (
            f"Determine whether there is a 'association' relationship between "
            f"'{entity1}' and '{entity2}' in the following sentence: '{sentence}'. "
            f"\n\nAnswer only with one of the following options:\n"
            f"(A) Yes - '{entity1}' and '{entity2}' have a **direct** association (causation, correlation, substitution, inclusion, or biological/chemical interaction).\n"
            f"(B) NoAssociation - The sentence explicitly states that '{entity1}' and '{entity2}' have **no effect** on each other, or '{entity1}' is **excluded from** '{entity2}'.\n"
            f"(C) Unrelated - '{entity1}' and '{entity2}' are **mentioned together** but without any meaningful connection.\n\n"
            f"**Guidelines for classification:**\n"
            f"- Select **(B) NoAssociation** if the sentence explicitly states that there is **no effect**, **no impact**, or **no causal relationship** between '{entity1}' and '{entity2}'.\n"
            f"- Select **(B) NoAssociation** if '{entity1}' is explicitly **excluded, bypassed, or omitted** from '{entity2}' (e.g., 'privation from', 'not included in', 'bypassing', 'X-free', 'avoids', etc.).\n"
            f"- Select **(C) Unrelated** if '{entity1}' and '{entity2}' are simply mentioned together but **without** any logical or meaningful relationship.\n\n"
            f"**Examples of when to answer (B) NoAssociation:**\n"
            f"- The sentence explicitly states that '{entity1}' **does not affect** '{entity2}' in any way.\n"
            f"- The sentence concludes that **there is no causal or meaningful relationship** between '{entity1}' and '{entity2}'.\n"
            f"- A study or analysis is mentioned, but it states **no significant association** between '{entity1}' and '{entity2}'.\n"
            f"- The sentence implies a possible connection but **ultimately dismisses it** as insignificant.\n"
            f"- '{entity1}' is **excluded, omitted, or bypassed** in relation to '{entity2}' (e.g., 'X-free diet excludes Y', 'X avoids Y', 'X is omitted from Y').\n\n"
            f"**Examples of when to answer (C) Unrelated:**\n"
            f"- '{entity1}' and '{entity2}' are mentioned **in the same list**, but there is **no stated relationship**.\n"
            f"- The sentence describes characteristics of '{entity1}' and '{entity2}' separately without linking them.\n"
            f"- '{entity1}' and '{entity2}' belong to the same category (e.g., both are chemicals, foods, or medical terms) but are not directly connected.\n"
            f"- The sentence compares '{entity1}' and '{entity2}' (e.g., one is healthier than the other) but **does not describe any direct interaction**.\n\n"
            f"**Answer (A) Yes ONLY IF:**\n"
            f"- The sentence explicitly states that '{entity1}' and '{entity2}' **have a direct association**, such as **causation, correlation, substitution, or biological/chemical interaction**.\n"
            f"- '{entity1}' and '{entity2}' are part of the **same biological or functional process** with an explicitly stated relationship.\n"
            f"- One entity is a **component, ingredient, or essential part** of the other.\n"
            f"- '{entity1}' and '{entity2}' directly affect each other in a **positive or negative direction**.\n\n"
            f"**Examples of when to answer (A) Yes:**\n"
            f"- '{entity1}' and '{entity2}' have a **direct or indirect effect** on each other.\n"
            f"- The sentence explicitly states that '{entity1}' and '{entity2}' are related through **substitution, correlation, or causation**.\n"
            f"- '{entity1}' and '{entity2}' are part of the same **biological/metabolic pathway** with a clear, explicit relationship.\n"
            f"- One entity is **necessary for the function of the other**.\n\n"
            f"**Provide your answer in the following format:**\n"
            f"Reason: Provide an explanation focusing on key verbs, negations, and the relationship between '{entity1}' and '{entity2}'.\n"
            f"Answer: (A) Yes OR (B) NoAssociation OR (C) Unrelated\n"
        )

        return system_prompt, user_prompt
