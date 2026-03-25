from ..classifier_base import ClassifierBase


class C5AssoCausal(ClassifierBase):
    def __init__(self, **kwargs):
        current_labels = ["CausalEffect", "Association"]
        current_label_to_label = {
            "(A) CausalEffect": "causalEffect",
            "(B) Association": "association",
        }
        super().__init__(current_labels, current_label_to_label, **kwargs)

    def prompt(self, entity1, entity2, sentence):
        system_prompt = (
            "You are a Clinical Scientific Reviewer specializing in biomedical relation extraction. "
            "Your task is to analyze given texts and extract meaningful relationships between biomedical entities. "
            "Provide accurate, concise, and well-structured explanations based on scientific principles."
        )

        user_prompt = (
            f"Determine whether the relationship between '{entity1}' and '{entity2}' in the following sentence is a **direct causal effect** or a **general association**: '{sentence}' "
            f"\n\nAnswer only with one of the following options:\n"
            f"(A) CausalEffect - '{entity1}' **directly causes, leads to, results in, or significantly alters** '{entity2}'.\n"
            f"(B) Association - '{entity1}' and '{entity2}' are related, but causation is **not explicitly stated**.\n\n"
            f"**Key distinctions:**\n"
            f"- Choose **(A) CausalEffect** if the sentence explicitly states that '{entity1}' **directly causes** '{entity2}'. "
            f"Common indicators: 'causes', 'results in', 'leads to', 'is responsible for', 'induces', 'triggers', 'generates', 'significantly alters', 'plays a key role in'.\n"
            f"- Choose **(B) Association** if '{entity1}' and '{entity2}' are linked but the direction of causality is not certain. "
            f"Common indicators: 'associated with', 'linked to', 'correlated with', 'influences', 'impacts', 'modifies'.\n\n"
            f"**Additional rule:**\n"
            f"- If the sentence **strongly implies a cause-effect relationship** without explicitly stating it, but the mechanism is known from scientific literature, choose **(A) CausalEffect**.\n"
            f"- If the sentence only states a statistical or observational link, choose **(B) Association**.\n"
            f"- If '{entity1}' is described as a **risk factor** for '{entity2}' it should be labeled as **(B) Association**.\n\n"
            f"**Examples:**\n"
            f"- (A) CausalEffect: 'Obesity **leads to** increased blood pressure.'\n"
            f"- (A) CausalEffect: 'High salt intake **induces** hypertension by increasing water retention.'\n"
            f"- (A) CausalEffect: 'A ketogenic diet **significantly alters** lipid metabolism.'\n"
            f"- (B) Association: 'Obesity **is a major risk factor** for hypertension.'\n"
            f"- (B) Association: 'Obesity **is associated with** increased blood pressure.'\n"
            f"- (B) Association: 'People who consume more salt **tend to** have higher blood pressure.'\n\n"
            f"Provide your answer in the following format:\n"
            f"Reason: Explain the relationship based on key verbs and causality.\n"
            f"Answer: (A) CausalEffect OR (B) Association\n"
        )

        return system_prompt, user_prompt
