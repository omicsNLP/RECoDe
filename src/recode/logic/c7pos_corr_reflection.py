from ..classifier_base import ClassifierBase


class C7PositiveCorrReflectionWithContains(ClassifierBase):
    """Reflection format for the PositiveCorrelation classification task."""

    def __init__(self, **kwargs):
        current_labels = [
            "Confirm to PositiveCorrelation",
            "Revise to IncreaseAssociation",
            "Revise to Association",
        ]
        current_label_to_label = {
            "(A) Confirm to PositiveCorrelation": "positiveCorrelation",
            "(B) Revise to IncreaseAssociation": "increaseAssociation",
            "(C) Revise to Association": "association",
        }
        self.predicted_label = "positiveCorrelation"
        super().__init__(current_labels, current_label_to_label, **kwargs)

    def prompt(self, entity1, entity2, sentence):
        system_prompt = (
            "You are a Clinical Scientific Reviewer specializing in biomedical relation extraction. "
            "Your task is to reflect on earlier decisions and validate if the classification is still valid."
        )

        user_prompt = (
            f"The previous analysis classified the relationship between '{entity1}' and '{entity2}' in the following sentence as **PositiveCorrelation**.\n\n"
            f"Sentence: {sentence}\n\n"
            f"Please re-evaluate the classification.\n\n"
            f"Choose one of the following options:\n"
            f"(A) Confirm to PositiveCorrelation - Both entities tend to **increase or decrease together**, either across time or under specific conditions or groups. "
            f"This includes phrases like 'correlated with' or 'positively correlated with'. "
            f"In cases such as 'positively associated with' or 'linked to higher levels of', choose this only **if both '{entity1}' and '{entity2}' are clearly described as increasing (e.g., 'high', 'higher', 'elevated') or decreasing (e.g., 'low', 'lower') in the same direction**. "
            f"Directionality can also be **implicit**, such as numeric trends (e.g., '10mg to 20mg'). "
            f"If such co-directional change is not clearly described for both entities, consider other options instead.\n\n"
            f"(B) Revise to IncreaseAssociation - One entity is observed or reported to be at a **higher level when the other is present**, even if causality is not explicitly stated. "
            f"This includes expressions such as '{entity1} was significantly higher in {entity2}', or numeric comparisons that suggest an increase (e.g., from 10mg to 20mg). "
            f"Also apply this when an increase is **implicitly indicated** through phrases like 'elevated levels', 'upregulated in group X', or similar comparative language.\n\n"
            f"(C) Revise to Association - The two are associated, but the **directional or co-variation pattern** is not clear enough.\n\n"
            f"**Guidelines:**\n"
            f"- Choose (A) if both entities clearly increase/decrease together (e.g., correlated or co-varying patterns).\n"
            f"- Choose (B) if one entity increases (or is higher) **in the presence or context of** the other.\n"
            f"- Choose (C) if the relationship is vague or lacks clear directionality.\n\n"
            f"**Examples:**\n"
            f"- (A) Confirm to PositiveCorrelation - 'Higher levels of X are associated with higher levels of Y.'\n"
            f"- (A) Confirm to PositiveCorrelation - '<entity1>Firmicutes/Bacteroidetes ratio</entity1> significantly correlated positively with <entity2>total cholesterol</entity2>.'\n"
            f"- (B) Revise to IncreaseAssociation - 'X increases Y levels significantly.'\n"
            f" -(B) Revise to IncreaseAssociation - 'The mean value of the <entity1>C-reactive protein</entity1> had the highly significant highest value among the <entity2>obese</entity2> women without MetS.'\n"
            f"- (C) Revise to Association - 'The <entity1>gut microbiota</entity1> is associated with <entity2>obesity</entity2>.'\n"
            f"\n"
            f"Provide your answer in the following format:\n"
            f"Reason: Briefly explain the reasoning using key phrases or structures in the sentence.\n"
            f"Answer: (A) Confirm to PositiveCorrelation OR (B) Revise to IncreaseAssociation OR (C) Revise to Association"
        )

        return system_prompt, user_prompt
