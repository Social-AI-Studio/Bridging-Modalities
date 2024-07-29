## Zero-Shot

### Prompt Templates
ZS_SINGLE_TURN_PROMPT = """## Definition of Hate Speech:
Hate speech is any communication that belittles, discriminates against, or incites violence against individuals or groups based on attributes such as race, religion, ethnicity, gender, sexual orientation, disability, or other distinguishing characteristics. This includes, but is not limited to, slurs, threats, dehumanizing language, and advocating for exclusion or violence against these individuals or groups.

## Instruction
Evaluate the content using the definition of hate speech to determine if it is considered as hateful. Respond with either "Hateful" or "Not Hateful."

## Task: Evaluate the following content and respond with either "Hateful" or "Not Hateful" based on the provided definition of hate speech.
Content: {content}
Answer:
"""

## Few-Shot

### Single-Turn Prompt Templates
INTRODUCTION = """## Definition of Hate Speech:
Hate speech is any communication that belittles, discriminates against, or incites violence against individuals or groups based on attributes such as race, religion, ethnicity, gender, sexual orientation, disability, or other distinguishing characteristics. This includes, but is not limited to, slurs, threats, dehumanizing language, and advocating for exclusion or violence against these individuals or groups.

## Instruction
Evaluate the content using the definition of hate speech to determine if it is considered as hateful. Respond with either "Hateful" or "Not Hateful."

## Demonstration Examples"""

EXAMPLE_TEMPLATE_WITH_RATIONALE = """### Example {index}
Content: {content}
Answer: {answer}
Rationale: {rationale}
"""

QUESTION_TEMPLATE = """## Task: Evaluate the following content and respond with either "Hateful" or "Not Hateful" based on the provided definition of hate speech.
Content: {content}
Answer: """


### Multi-Turn Prompt Templates
QUESTION_MULTI_TURN_TEMPLATE = """Content: {content}"""
ANSWER_MULTI_TURN_TEMPLATE = """Answer: {answer}
Rationale: {rationale}"""

### SYSTEM PROMPTS
SYSTEM_PROMPT = """You are an AI chatbot assistant that evaluate the provided content using the hate speech definition to determine if it is considered as hateful. Respond with either "Hateful" or "Not Hateful."

## Hate Speech Definition:
Hate speech is any communication that belittles, discriminates against, or incites violence against individuals or groups based on attributes such as race, religion, ethnicity, gender, sexual orientation, disability, or other distinguishing characteristics. This includes, but is not limited to, slurs, threats, dehumanizing language, and advocating for exclusion or violence against these individuals or groups.
"""

INTRODUCTION_WITHOUT_INSTRUCTIONS = """## Demonstration Examples"""