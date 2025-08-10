#import the vectorized query to LLM
from embedding import do_embedding
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

ans, query = do_embedding()

#create the context
contextText = "\n\n".join(doc.page_content for doc in ans)


#create a template

template = """""
You are an assistant answering questions based on a provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:

"""
#initilize the LLM model

model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

def llm_response():
    prompt = PromptTemplate(input_variables=['context', 'question'],
                            template=template)

    #finalized prompt

    final_prompt = prompt.format(context = contextText, question = query)

    response = model.predict(final_prompt)
    

    return response

test = llm_response()

print(test)


