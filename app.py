from flask import Flask, render_template, request, jsonify
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

app = Flask(__name__)

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token="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",
                     repo_id=repo_id,
                     model_kwargs={"temperature": 0.7, "max_new_tokens": 500})

template = """
You are a helpful AI assistant and provide the answer for the question asked politely.

{question}
"""

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat_start', methods=['POST'])
def chat_start():
    question = request.form.get('question')

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    response = llm_chain.run(question)
    return jsonify(response=response)


if __name__ == "__main__":
    app.run()
