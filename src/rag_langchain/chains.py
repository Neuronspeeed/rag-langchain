from typing import Literal
import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from .config import OPENAI_API_KEY

from rag_langchain.prompts import (
    rag_prompt,
    db_query_rewrite_prompt,
    hallucination_prompt,
    answer_prompt,
    query_feedback_prompt,
    generation_feedback_prompt,
    give_up_prompt,
    grade_doc_prompt,
    knowledge_extraction_prompt,
    router_prompt,
    websearch_query_rewrite_prompt,
    simple_question_prompt
)



class GradeHallucinations(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class RouteQuery(BaseModel):
    route: Literal["vectorstore", "websearch", "QA_LM"] = Field(
        description="Given a user question choose to route it to web search (websearch), a vectorstore (vectorstore), or a QA language model (QA_LM).",
    )

llm_engine = ChatOpenAI(model='gpt-4o-mini', openai_api_key=OPENAI_API_KEY)

print(f"Using OpenAI API Key in chains.py: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}")

try:
    rag_chain = rag_prompt | llm_engine | StrOutputParser()
except AuthenticationError:
    logger.error("Authentication failed. Please check your OpenAI API key.")
    raise
except RateLimitError:
    logger.error("OpenAI API rate limit exceeded. Please try again later.")
    raise
except Exception as e:
    logger.error(f"An unexpected error occurred: {str(e)}")
    raise

db_query_rewriter = db_query_rewrite_prompt | llm_engine | StrOutputParser()
hallucination_grader = hallucination_prompt | llm_engine.with_structured_output(GradeHallucinations)
answer_grader = answer_prompt | llm_engine.with_structured_output(GradeAnswer)
query_feedback_chain = query_feedback_prompt | llm_engine | StrOutputParser()
generation_feedback_chain = generation_feedback_prompt | llm_engine | StrOutputParser()
give_up_chain = give_up_prompt | llm_engine | StrOutputParser()
retrieval_grader = grade_doc_prompt | llm_engine.with_structured_output(GradeDocuments)
knowledge_extractor = knowledge_extraction_prompt | llm_engine | StrOutputParser()
question_router = router_prompt | llm_engine.with_structured_output(RouteQuery)
websearch_query_rewriter = websearch_query_rewrite_prompt | llm_engine | StrOutputParser()
simple_question_chain = simple_question_prompt | llm_engine | StrOutputParser()