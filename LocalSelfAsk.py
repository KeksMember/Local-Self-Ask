import requests
import json
from llama_cpp import Llama

prompt = ['''Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali 

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball 

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Are follow up questions needed here: Yes. 
Follow up: Who is the director of Jaws? 
Intermediate Answer: The director of Jaws is Steven Spielberg. 
Follow up: Where is Steven Spielberg from? 
Intermediate Answer: The United States. 
Follow up: Who is the director of Casino Royale? 
Intermediate Answer: The director of Casino Royale is Martin Campbell. 
Follow up: Where is Martin Campbell from? 
Intermediate Answer: New Zealand. 
So the final answer is: No

Question: ''',
'''
Are follow up questions needed here:''', ]

#LLM methods
modelConfig = dict(
        n_threads=16,
        n_ctx=2048,
        seed=0,
        verbose=False
)

model = Llama(model_path="Models/gpt4-x-vicuna-13B.ggml.q5_1.bin", **modelConfig)

def getMainLLMResponse(prompt, stop):
    modelOutputConfig = dict(
        max_tokens=1024,
        top_k=64,
        top_p=0.45,
        stop=[stop],
        echo=False
    )

    outputLlama = str(model(prompt, **modelOutputConfig))
    textOutputStart = outputLlama.find("'text':") + 9
    textOutputEnd = outputLlama.find("index") - 4
    textOutput = outputLlama[textOutputStart:textOutputEnd]
    return textOutput

def curateContent(input, question):
    modelOutputConfig = dict(
        max_tokens=1024,
        top_k=64,
        top_p=0.1,
        echo=False
    )

    inputPrompt = """Your task is to find an answer to the given question inside the given text samples. Be intuitive and clever. Question: {}\n\n""".format(question)
    for element in input:
        inputPrompt += "Sample: " + element + "\n\n"
    inputPrompt += "Answer: "

    outputLlama = str(model(inputPrompt, **modelOutputConfig))
    textOutputStart = outputLlama.find("'text':") + 9
    textOutputEnd = outputLlama.find("index") - 4
    textOutput = outputLlama[textOutputStart:textOutputEnd]
    return textOutput

# web search
def search(query):
    raw = returnRawSearchResult(query)
    rawJSON = json.loads(raw)

    if rawJSON["answers"] != "[]":
        print("###Method: search, Answer found!")
        answer = rawJSON["answers"]
        return curateContent(answer, query)
    else:
        print("###Method: search, Answer not found!")
        resultList = rawJSON["results"]
        contentList = []

        for result in resultList:
            resultJSON = json.loads(json.dumps(result))
            contentList.append(resultJSON["content"])

        finalAnswer = curateContent(contentList, query)
        return finalAnswer

def returnRawSearchResult(query):
    params = {'q': query, 'format': 'json'}

    response = requests.get("http://localhost:2000/search", params=params)
    return response.text

# util
def reformat(input):
    return input.replace("\\n", "\n")

def getQuery(input):
    if '\\n' in input:
        return input.split("\\nFollow up:")[-1]
    else:
        return input

# main
def run(question):
    activePrompt = prompt[0] + question + prompt[1]

    response = getMainLLMResponse(activePrompt, "\nIntermediate answer: ")

    while "Follow up" in response:
        query = getQuery(response)

        searchEngineAnswer = search(query)

        activePrompt += reformat(response)

        if searchEngineAnswer is not None:
            activePrompt += "\nIntermediate answer: " + searchEngineAnswer
            response = getMainLLMResponse(activePrompt, "\nIntermediate answer:")
        else:
            print("Search failed.")

    if "\\nSo the final answer is:" not in response:
        activePrompt += "\nSo the final answer is:"
        response = getMainLLMResponse(activePrompt, "\n")

    return reformat(activePrompt) + response


print(run("What is the main battle tank of the NSO faction in the game Planetside 2 called?"))