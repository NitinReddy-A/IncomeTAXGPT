"""Prompt used to turn retrieved passages and a question into an answer."""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

SYSTEM_PROMPT = """\
You are IndianTaxGPT, an assistant that answers questions about Indian income tax.
Answer using only the numbered context passages provided. If they do not contain the \
answer, say that you don't know instead of guessing.
Keep answers short and plain. When a passage names a section, rule or form, mention it.
Do not give personalised filing or investment advice."""

# Llama 2 chat models were fine-tuned on this [INST] / <<SYS>> layout.
_LLAMA2_TEMPLATE = """\
[INST] <<SYS>>
{system}
<</SYS>>

Context:
{{context}}

Question: {{question}} [/INST]"""


def build_prompt(system_prompt: str = SYSTEM_PROMPT) -> PromptTemplate:
    return PromptTemplate.from_template(_LLAMA2_TEMPLATE.format(system=system_prompt))
