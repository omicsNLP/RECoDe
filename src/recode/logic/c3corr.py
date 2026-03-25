from ..classifier_base import ClassifierBase


class C3Corr(ClassifierBase):
    def __init__(self, **kwargs):
        current_labels = ["PositiveCorrelation", "NegativeCorrelation"]
        current_label_to_label = {
            "(A) PositiveCorrelation": "positiveCorrelation",
            "(B) NegativeCorrelation": "negativeCorrelation",
        }
        super().__init__(current_labels, current_label_to_label, **kwargs)

    def prompt(self, entity1, entity2, sentence):
        system_prompt = (
            "You are a Clinical Scientific Reviewer specializing in biomedical relation extraction. "
            "Your task is to analyze given texts and extract meaningful relationships between biomedical entities. "
            "Provide accurate, concise, and well-structured explanations based on scientific principles."
        )

        user_prompt = (
            f"Determine the **specific type of correlation** between '{entity1}' and '{entity2}' in the following sentence: '{sentence}'. "
            f"\n\nAnswer only with one of the following options:\n"
            f"(A) PositiveCorrelation - '{entity1}' and '{entity2}' increase or decrease together.\n"
            f"(B) NegativeCorrelation - When '{entity1}' increases, '{entity2}' decreases (or vice versa).\n\n"
            f"**Key distinctions:**\n"
            f"- If '{entity1}' and '{entity2}' increase or decrease together, choose **(A) PositiveCorrelation**.\n"
            f"- If '{entity1}' increases while '{entity2}' decreases (or vice versa), choose **(B) NegativeCorrelation**.\n\n"
            f"**Examples:**\n"
            f"- (A) PositiveCorrelation: 'Higher X intake is associated with higher Y levels.'\n"
            f"- (B) NegativeCorrelation: 'As X levels increase, Y levels decrease.'\n\n"
            f"Provide your answer in the following format:\n"
            f"Reason: Explain the relationship based on key verbs and directionality.\n"
            f"Answer: (A) PositiveCorrelation OR (B) NegativeCorrelation\n"
        )

        return system_prompt, user_prompt
