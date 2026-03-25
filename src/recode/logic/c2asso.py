from ..classifier_base import ClassifierBase


class C2Asso(ClassifierBase):
    def __init__(self, **kwargs):
        current_labels = ["Correlation", "Association", "Contains"]
        current_label_to_label = {
            "(A) Correlation": "MiddleCorrelation",
            "(B) Association": "MiddleAssociation",
            "(C) Contains": "Contains",
        }
        super().__init__(current_labels, current_label_to_label, **kwargs)

    def prompt(self, entity1, entity2, sentence):
        system_prompt = (
            "You are a Clinical Scientific Reviewer specializing in biomedical relation extraction. "
            "Your task is to analyze given texts and extract meaningful relationships between biomedical entities. "
            "Provide accurate, concise, and well-structured explanations based on scientific principles."
        )

        user_prompt = (
            f"Determine the broad category of relationship between '{entity1}' and '{entity2}' in the following sentence: '{sentence}'. "
            f"\n\nAnswer only with one of the following options:\n"
            f"(A) Correlation - '{entity1}' and '{entity2}' show a statistical or directional relationship (e.g., positive or negative correlation).\n"
            f"(B) Association - '{entity1}' and '{entity2}' have a direct connection (e.g., causal effect, increase, decrease, or influence).\n"
            f"(C) Contains - '{entity1}' contains, includes, or substitutes '{entity2}'.\n\n"
            f"Guidelines for classification:\n"
            f"- (A) Correlation: Use this if '{entity1}' and '{entity2}' are correlated (either positively or negatively) but without direct influence.\n"
            f"- (B) Association: Use this if '{entity1}' has a direct effect on '{entity2}', increases, decreases, or influences it in any way.\n"
            f"- (C) Contains: Use this if '{entity1}' consists of, includes, or substitutes '{entity2}'. If '{entity1}' replaces or is replaced by '{entity2}', use Contains.\n\n"
            f"**Key distinctions:**\n"
            f"- Choose **(B) Association** if '{entity1}' leads to or implies a higher/lower level or likelihood of '{entity2}', or vice versa ('{entity2}' to '{entity1}'), even if the sentence uses passive voice or indirect expressions.\n"
            f"- Choose **(A) Correlation** only when the sentence mentions a co-occurrence or relationship between '{entity1}' and '{entity2}' without implying any directional effect.\n"
            f"- Choose **(C) Contains** if one entity includes, consists of, or substitutes the other.\n\n"
            f"- If the sentence explicitly states that '{entity1}' and '{entity2}' are correlated but does not specify causation, choose **(A) Correlation**.\n"
            f"- If '{entity1}' is said to influence, increase, or decrease '{entity2}', choose **(B) Association**.\n"
            f"- If '{entity1}' is replacing, being replaced by, or is a substitute for '{entity2}', choose **(C) Contains**.\n\n"
            f"Examples:\n"
            f"- (A) Correlation: 'Higher X levels are associated with higher/lower Y levels.' OR 'When X levels are low, some gene increases Y levels'\n"
            f"- (B) Association: 'X causes Y' OR 'X increases/decreases Y' OR 'X supplementation leads to a reduction in Y levels.' OR 'X had a higher prevalence of B'.\n"
            f"- (C) Contains: 'X is made up of Y' OR 'X is a substitute for Y' OR 'X is replaced with Y'.\n\n"
            f"Provide your answer in the following format:\n"
            f"Reason: Explain the relationship based on key verbs and negations if any.\n"
            f"Answer: (A) Correlation OR (B) Association OR (C) Contains\n"
        )

        return system_prompt, user_prompt
