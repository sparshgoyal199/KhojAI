# judge.py
from deepeval.models import DeepEvalBaseLLM
from core.llm import llm  # your existing ChatGroq(...) instance


class GroqJudge(DeepEvalBaseLLM):
    """Adapter that lets DeepEval use the existing ChatGroq as a judge."""

    def __init__(self, llm_instance=llm, model_name: str | None = None):
        self._llm = llm_instance
        # ChatGroq stores the model id on .model_name; fall back gracefully
        self._model_name = model_name or getattr(llm_instance, "model_name", "groq-judge")

    def load_model(self):
        return self._llm

    def get_model_name(self) -> str:
        return self._model_name

    # ---- text generation ----
    def generate(self, prompt: str, schema=None):
        if schema is not None:
            structured = self._llm.with_structured_output(schema)
            return structured.invoke(prompt)
        return self._llm.invoke(prompt).content

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            structured = self._llm.with_structured_output(schema)
            return await structured.ainvoke(prompt)
        result = await self._llm.ainvoke(prompt)
        return result.content


# Import this singleton in your eval script
judge = GroqJudge()