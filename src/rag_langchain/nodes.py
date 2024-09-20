from typing import List, Literal, Optional
from pydantic import BaseModel
from rag_langchain.data_index import retriever
from .config import logger, OPENAI_API_KEY, TAVILY_API_KEY

from rag_langchain.chains import (
    rag_chain,
    db_query_rewriter,
    hallucination_grader,
    answer_grader,
    generation_feedback_chain,
    query_feedback_chain,
    retrieval_grader,
    knowledge_extractor,
    question_router,
    simple_question_chain,
    give_up_chain,
    websearch_query_rewriter
)

from rag_langchain.websearch import web_search_tool

MAX_RETRIEVALS = 3
MAX_GENERATIONS = 3
MAX_TOTAL_STEPS = 50

class GraphState(BaseModel):
    question: Optional[str] = None
    generation: Optional[str] = None
    documents: List[str] = []
    rewritten_question: Optional[str] = None
    query_feedbacks: List[str] = []
    generation_feedbacks: List[str] = []
    generation_num: int = 0
    retrieval_num: int = 0
    search_mode: str = "QA_LM"
    total_steps: int = 0

def retriever_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    new_documents = retriever.invoke(state.rewritten_question)
    new_documents = [d.page_content for d in new_documents]
    if not new_documents:
        logger.info("No new documents retrieved.")
    state.documents.extend(new_documents)
    logger.info(f"Retrieved {len(new_documents)} new documents.")
    return {
        "documents": state.documents, 
        "retrieval_num": state.retrieval_num + 1
    }

def generation_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    generation = rag_chain.invoke({
        "context": "\n\n".join(state.documents), 
        "question": state.question, 
        "feedback": "\n".join(state.generation_feedbacks)
    })
    logger.info(f"Generated answer: {generation}")
    return {
        "generation": generation,
        "generation_num": state.generation_num + 1
    }

def db_query_rewriting_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    rewritten_question = db_query_rewriter.invoke({
        "question": state.question,
        "feedback": "\n".join(state.query_feedbacks)
    })
    logger.info(f"Rewritten Question: {rewritten_question}")
    return {"rewritten_question": rewritten_question, "search_mode": "vectorstore"}

def answer_evaluation_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    hallucination_grade = hallucination_grader.invoke({
        "documents": state.documents,
        "generation": state.generation
    })
    logger.info(f"Hallucination Grade: {hallucination_grade.binary_score}")

    if hallucination_grade.binary_score == "yes":
        answer_grade = answer_grader.invoke({
            "question": state.question,
            "generation": state.generation
        })
        logger.info(f"Answer Grade: {answer_grade.binary_score}")

        if answer_grade.binary_score == "yes":
            return "useful"

    # Handle cases where the answer is not useful
    if state.generation_num >= MAX_GENERATIONS:
        logger.warning("Maximum generations reached without a useful answer.")
        return "max_generation_reached"
    else:
        logger.info("Answer not useful. Triggering feedback loops.")
        return "not relevant"

def generation_feedback_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    feedback = generation_feedback_chain.invoke({
        "question": state.question,
        "documents": "\n\n".join(state.documents),
        "generation": state.generation
    })

    feedback = 'Feedback about the answer "{}": {}'.format(
        state.generation, feedback
    )
    state.generation_feedbacks.append(feedback)
    logger.info(f"Generation Feedback: {feedback}")
    return {"generation_feedbacks": state.generation_feedbacks}

def query_feedback_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    feedback = query_feedback_chain.invoke({
        "question": state.question,
        "rewritten_question": state.rewritten_question,
        "documents": "\n\n".join(state.documents),
        "generation": state.generation
    })

    feedback = 'Feedback about the query "{}": {}'.format(
        state.rewritten_question, feedback
    )
    state.query_feedbacks.append(feedback)
    logger.info(f"Query Feedback: {feedback}")
    return {"query_feedbacks": state.query_feedbacks}

def give_up_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    response = give_up_chain.invoke(state.question)
    logger.info(f"Give Up Response: {response}")
    return {"generation": response, "current_step": "END"}

def filter_relevant_documents_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    # first, we grade every documents
    grades = retrieval_grader.batch([
        {"question": state.question, "document": doc} 
        for doc in state.documents
    ])
    # Then we keep only the documents that were graded as relevant
    filtered_docs = [
        doc for grade, doc 
        in zip(grades, state.documents) 
        if grade.binary_score == 'yes'
    ]

    # If we didn't get any relevant document, let's capture that 
    # as a feedback for the next retrieval iteration
    if not filtered_docs:
        feedback = 'Feedback about the query "{}": did not generate any relevant documents.'.format(
            state.rewritten_question
        )
        state.query_feedbacks.append(feedback)
        logger.info(f"No relevant documents found. Added feedback: {feedback}")

    logger.info(f"Filtered {len(filtered_docs)} relevant documents.")
    return {
        "documents": filtered_docs, 
        "query_feedbacks": state.query_feedbacks
    }

def knowledge_extractor_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    filtered_docs = knowledge_extractor.batch([
        {"question": state.question, "document": doc} 
        for doc in state.documents
    ])
    # we keep only the non empty documents
    filtered_docs = [doc for doc in filtered_docs if doc]
    logger.info(f"Extracted {len(filtered_docs)} knowledge documents.")
    return {"documents": filtered_docs}

def router_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    route_query = question_router.invoke(state.question)
    logger.info(f"Routed to: {route_query.route}")
    return route_query.route

def simple_question_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    answer = simple_question_chain.invoke(state.question)
    logger.info(f"Simple Question Answer: {answer}")
    return {"generation": answer, "search_mode": "QA_LM"}

def websearch_query_rewriting_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    rewritten_question = websearch_query_rewriter.invoke({
        "question": state.question, 
        "feedback": "\n".join(state.query_feedbacks)
    })
    if state.search_mode != "websearch":
        state.retrieval_num = 0    
    logger.info(f"Rewritten Websearch Question: {rewritten_question}")
    return {
        "rewritten_question": rewritten_question, 
        "search_mode": "websearch",
        "retrieval_num": state.retrieval_num
    }

def web_search_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    new_docs = web_search_tool.invoke(
        {"query": state.rewritten_question}
    )
    web_results = [d["content"] for d in new_docs]
    state.documents.extend(web_results)
    logger.info(f"Retrieved {len(web_results)} web search results.")
    return {
        "documents": state.documents, 
        "retrieval_num": state.retrieval_num + 1
    }

def search_mode_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    logger.info(f"Current Search Mode: {state.search_mode}")
    return state.search_mode

def relevant_documents_validation_node(state: GraphState):
    state.total_steps += 1
    if state.total_steps > MAX_TOTAL_STEPS:
        logger.warning("Maximum total steps exceeded. Triggering give_up.")
        return "give_up"
    
    if state.documents:
        logger.info("Documents available. Proceeding to knowledge extraction.")
        return "knowledge_extraction"
    elif state.search_mode == 'vectorstore' and state.retrieval_num > MAX_RETRIEVALS:
        logger.info("Max DB search attempts reached. Switching to websearch.")
        return "max_db_search"
    elif state.search_mode == 'websearch' and state.retrieval_num > MAX_RETRIEVALS:
        logger.info("Max websearch attempts reached. Giving up.")
        return "max_websearch"
    else:
        logger.info(f"Continuing with current search mode: {state.search_mode}")
        return state.search_mode


