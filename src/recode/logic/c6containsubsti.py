from ..classifier_base import ClassifierBase


class C6ContainSubsti(ClassifierBase):
    def __init__(self, **kwargs):
        current_labels = ["Consists", "Substitution"]
        current_label_to_label = {
            "(A) Consists": "consists",
            "(B) Substitution": "substitution",
        }
        super().__init__(current_labels, current_label_to_label, **kwargs)

    def prompt(self, entity1, entity2, sentence):
        system_prompt = (
            "You are a Clinical Scientific Reviewer specializing in biomedical relation extraction. "
            "Your task is to analyze given texts and extract meaningful relationships between biomedical entities. "
            "Provide accurate, concise, and well-structured explanations based on scientific principles."
        )

        user_prompt = (
            f"Determine the **specific type of 'contains' relationship** between '{entity1}' and '{entity2}' in the following sentence: '{sentence}'. "
            f"\n\nAnswer only with one of the following options:\n"
            f"(A) Consists - '{entity1}' **is made up of, includes, or contains** '{entity2}'.\n"
            f"(B) Substitution - '{entity1}' **replaces or is replaced by** '{entity2}'.\n\n"
            f"**Key distinctions:**\n"
            f"- Choose **(A) Consists** if '{entity1}' contains '{entity2}' as an ingredient, component, or part.\n"
            f"- Choose **(B) Substitution** if '{entity1}' and '{entity2}' are alternatives to each other, meaning one replaces or is replaced by the other.\n\n"
            f"**Additional rule:**\n"
            f"- If the sentence **describes '{entity1}' as containing '{entity2}'**, such as an ingredient in food, a molecule in a compound, or a component in a system, choose **(A) Consists**.\n"
            f"- If the sentence **describes '{entity1}' being swapped or exchanged for '{entity2}'**, such as in dietary replacements or drug alternatives, choose **(B) Substitution**.\n\n"
            f"**Examples:**\n"
            f"- (A) Consists: 'Milk **contains** calcium.'\n"
            f"- (A) Consists: 'Fish oil **is rich in** omega-3 fatty acids.'\n"
            f"- (A) Consists: 'Wheat flour **is a source of** gluten.'\n"
            f"- (B) Substitution: 'Butter **can be replaced with** margarine in this recipe.'\n"
            f"- (B) Substitution: 'Plant-based protein **is an alternative to** animal protein.'\n"
            f"- (B) Substitution: 'Artificial sweeteners **substitute for** sugar in many processed foods.'\n\n"
            f"Provide your answer in the following format:\n"
            f"Reason: Explain the relationship based on key verbs and the nature of the relationship.\n"
            f"Answer: (A) Consists OR (B) Substitution\n"
        )

        return system_prompt, user_prompt
