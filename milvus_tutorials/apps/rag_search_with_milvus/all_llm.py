def get_llm_answer(llm_model, context: str, question: str):
    messages = [
        (
            "system",
            context
        ),
        ("human", question),
    ]
    ai_msg = llm_model.invoke(messages)    
    return ai_msg.content