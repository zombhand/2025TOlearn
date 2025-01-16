# 成为 LLM 工程师的学习经历

# 使用非官网的方法

MODEL

openai = OpenAI(base_url=, api_key=)

# 遇到的软件安装

ffmpeg安装先下载源码

然后使用`choco install ffmpeg-full`进行编译

# week1

## day1 start

安装了ANACONDA 是一个用于数据科学和机器学习的开源平台，主要包含大量的工具、库和包，帮助开发者快速搭建数据分析、科学计算和人工智能相关的开发环境。它是基于 Python 和 R 的软件发行版，提供了一个完整的生态系统，用于简化环境管理和包管理。

conda env create -f environment.yml

env是环境

conda activate llms



python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

jupyter lab（进入一个工作站）

创建DOT ENV文件，这是以您可以访问的方式存储秘密的常见方法

一定要是.env文件

win + R 输入 notepad

OPENAI_API_KEY=sk-

```
(base) PS C:\Users\29326> cd ed
(base) PS C:\Users\29326\ed> cd .\projects\
(base) PS C:\Users\29326\ed\projects> cd .\llm_engineering-main\
(base) PS C:\Users\29326\ed\projects\llm_engineering-main> conda activate llms
(llms) PS C:\Users\29326\ed\projects\llm_engineering-main> jupyter lab
```

```
imports

import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI

If you get an error running this cell, then please head over to the troubleshooting notebook!

```

## Connecting to OpenAI

```
# Load environment variables in a file called .env

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Check the key

if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not api_key.startswith(""):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("API key found and looks good so far!")
```

Models like GPT4o have been trained to receive instructions in a particular way.

They expect to receive:

**A system prompt** that tells them what task they are performing and what tone they should use

解释此对话的上下文内容，告诉他们执行什么样的任务，应该使用什么音调

**A user prompt** -- the conversation starter that they should reply to

是实际对话本身



The API from OpenAI expects to receive messages in a particular structure. Many of the other APIs share this structure:

```
解释[
    {"role": "system", "content": "system message goes here"},
    {"role": "user", "content": "user message goes here"}
]

To give you a preview, the next 2 cells make a rather simple call - we won't stretch the might GPT (yet!)
```

the API for OpenAI is very simple!

```python
# And now: call the OpenAI API. You will get very familiar with this!

def summarize(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages_for(website)
    )
    return response.choices[0].message.content
```

```python
# A function to display this nicely in the Jupyter output, using markdown

def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))
```

display_summary("https://edwarddonner.com") 总结网站



总的流程是

```python
# Step 1: Create your prompts

system_prompt = "something here"
user_prompt = """
    Lots of text
    Can be pasted here
"""

# Step 2: Make the messages list

messages = [] # fill this in

# Step 3: Call OpenAI

response =

# Step 4: print the result

print(
```



## day1总结

what can i do ALREADY

- Use Ollama to run LLMs locally on your box
- Write code to call OpenAl's frontier models
- Distinguish between the System and User prompt
- Summarization - applicable to many commercial problems

## day2 start

**Benefits:**

1. No API charges - open-source
2. Data doesn't leave your box

**Disadvantages:**

1. Significantly less power than Frontier Model

Big frontier models

**Closed-Source Frontier**

- GPT form OpenAI
- Claude from Anthropic
- Gemini from Google
- Command R from Cohere
- Perplexity

**Open-Source Frontier**

- Liama from Meta
- Mixtral from Mistral
- Qwen from Alibaba Cloud
- Gemma from Google
- Phi from Microsoft

Three ways to use models

- Chat interfaces
  - Like ChatGPT
- Cloud APIs
  - LLM API
  - Frameworks like LangChain
  - Managed AI cloud services
    - .Amazon Bedrock
    - .Google Vertex
    - .Azure ML
- Direct inference
  - With the HuggingFace Transformers library
  - With Ollama to run locally

Once complete, the ollama server should already be running locally.
If you visit:
http://localhost:11434/

You should see the message `Ollama is running`.

If not, bring up a new Terminal (Mac) or Powershell (Windows) and enter `ollama serve`
And in another Terminal (Mac) or Powershell (Windows), enter `ollama pull llama3.2`
Then try http://localhost:11434/ again.

If Ollama is slow on your machine, try using `llama3.2:1b` as an alternative. Run `ollama pull llama3.2:1b` from a Terminal or Powershell, and change the code below from `MODEL = "llama3.2"` to `MODEL = "llama3.2:1b"`

使用ollama

```python
# imports

import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display


# Constants

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"

# Create a messages list using the same format that we used for OpenAI

messages = [
    {"role": "user", "content": "Describe some of the business applications of Generative AI"}
]

payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }

# Let's just make sure the model is loaded

!ollama pull llama3.2

response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
print(response.json()['message']['content'])



```

## Introducing the ollama package

And now we'll do the same thing, but using the elegant ollama python package instead of a direct HTTP call.

Under the hood, it's making the same call as above to the ollama server running at localhost:11434

```python
import ollama

response = ollama.chat(model=MODEL, messages=messages)
print(response['message']['content'])
```

```python
// 制作一个网站总结具体流程-使用本地模型ollama
# imports

import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
import ollama

# Constants

MODEL = "llama3.2"

#A class to represent a Webpage
class Website:
    """
    A utility class to represent a Website that we have scraped
    是一个工具类，用来表示一个已经被抓取（scraped）的网页.
    """
    url: str
    title: str
    text: str
    
    def _init_(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        接收一个url的参数，代表要抓取的网页地址
        在初始化时，会根据URl去抓取网页内容，提取标题和纯文本
        """
        self.url = url 
        response = reponse.get(url) //发送HTTP请求，获取网页的内容
        soup = BeautifulSoup(response.content, 'htlm.parser')
        //一个常用的HTML解析库，解析响应的内容
        self.title = soup.title.string if soup.title else "NO title found"
        // 清理无关并提取文字
        for irrelevant in soup.body(["script", "style","img","input"]):
            irrelevant.decompose()
        self.test = soup.body.get_text(separator="\n", strip=True)
        # Let's try one out
        ed = Website("https://edwarddonner.com")
        print(ed.title)
        print(ed.text)
```

## Types of prompts

You may know this already - but if not, you will get very familiar with it!

Models like GPT4o have been trained to receive instructions in a particular way.

They expect to receive:

**A system prompt** that tells them what task they are performing and what tone they should use

**A user prompt** -- the conversation starter that they should reply to

```python
# Define our system prompt - you can experiment with this later, changing the last sentence to 'Respond in markdown in Spanish."

system_prompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."

# A function that writes a User Prompt that asks for summaries of websites:

def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "The contents of this website is as follows; \
please provide a short summary of this website in markdown. \
If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

# See how this function creates exactly the format above

def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]
# And now: call the Ollama function instead of OpenAI

def summarize(url):
    website = Website(url)
    messages = messages_for(website)
    response = ollama.chat(model=MODEL, messages=messages)
    return response['message']['content']

summarize("https://edwarddonner.com")

#A function to display this nicely in the Jupyter output, using mardonwn

def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))

display_summary("https://edwarddonner.com")    
```



## day2总结

What can i do ALREADY

- Understand the progression to become proficient
- Hava a first impression of the frontier LLMS
- User OpenAI API and Ollama to make summaries

## day3 start

**Frontier models and their end-user UIs**

![image-20250104094320440](C:\Users\29326\AppData\Roaming\Typora\typora-user-images\image-20250104094320440.png)

**Mind-blowing performance from Frontier LLMs**

- **Synthesizing information**(综合信息)

  Answering a question in depth with a structured,well researched answer and often including a summary

- **Fleshing out a skeleton**

  From a couple of notes,building out a well crafted email,or a blog post,and iterating on it with you until perfect

- **Coding**

  The ability to write and debug code is remarkable;far overtaken Stack Overflow as the resource for engineers

**LIMITATIONS OF FRONTIERMODELS(前沿模型的局限性)**

- **Specialized domains**(专业领域)

  Most are not PhD level,but closing in

- **Recent events**(近期事件)

  Limited knowledge beyond training cut-off date

- **Can confidently make mistakes**(能够自信的犯错误（幻觉）)

  Some curious blindspots

**In conclusion**

- **All 6 Frontier LLMs are shockingly good**

  Particularly at suynthesizing information and generating nuanced answers

- Claude tends to be favored by practitioners

  More humorous, more attention to safety,more concise

- **As they converge in capability,price may become the differentiator**

  Recent innovations have focuesd on lower cost variants

## day3总结

What can i do ALREADY

- Write code to call OpenAI's frontier models & summarize
- Explain the strengths and limitations of Frontier LLMs
- Compare and contrast the leading 6 models

## day4 start

**Along the way**

- Prompt Engineers

  The rise(and fall?)

- Custom GPTs

  and the GPT Store

- Copilots

  like MS Copilot and Github Copilot

- Agentization

  like Github Copilot Workspace

Introducing Tokens

- In the early days, neural networks were trained at the character level

  - Predict(预知) the next character in this sequence(顺序)

    Small vocab, but expects too much from the network

- Then neural networks were trained off words

  - Predict the next word in this sequence

    Much easier to learn from,but leads to enormous(巨大的) vocabs with rare(稀有的) words omitted

- The breakthrough was to work with chunks of  words,called 'tokens'

  A middle ground: manageble(可管理的) vocab, and useful information for the neural network Inaddition,elegantly handles word stems

GPT提供从视觉上查看该文本如何成为令牌(tokens)

单词之间的断裂在令牌化时也有意义

https://platform.openai.com/tokenizer



Rule-of-thumb: in typical English

writing:

- 1 token is ~4 characters
- 1 token is ~0.75 words
- So 1,000 tokens is ~750 words

## day4总结

What can i do ALREADY

- Write code to call OpenAI and Ollama & summarize
- Contrast the leading 6 Frontier LLMs
- Discuss transformers,tokens,context windows, API costs and more!

## day 5 start

```python
# A class to represent a Webpage

# Some websites need you to use proper headers when fetching them:
headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    """
    A utility class to represent a Website that we have scraped, now with links
    """

    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"
```

### First step: Have GPT-4o-mini figure out which links are relevant

### Use a call to gpt-4o-mini to read the links on a webpage, and respond in structured JSON.

It should decide which links are relevant, and replace relative links such as "/about" with "https://company.com/about".
We will use "one shot prompting" in which we provide an example of how it should respond in the prompt.

This is an excellent use case for an LLM, because it requires nuanced understanding. Imagine trying to code this without LLMs by parsing and analyzing the webpage - it would be very hard!

Sidenote: there is a more advanced technique called "Structured Outputs" in which we require the model to respond according to a spec. We cover this technique in Week 8 during our autonomous Agentic AI project.

```python
link_system_prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company, \
such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
link_system_prompt += "You should respond in JSON as in this example:"
link_system_prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page": "url": "https://another.full.url/careers"}
    ]
}
"""

def get_links_user_prompt(website):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt


def get_links(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(website)}
      ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)
```



**With a small adjustment, we can change this so that the results stream back from OpenAI, with the familiar typewriter animation**

```python
def stream_brochure(company_name, url):
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
          ],
        stream=True
    )
    
    response = ""
    display_handle = display(Markdown(""), display_id=True)
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        response = response.replace("```","").replace("markdown", "")
        update_display(Markdown(response), display_id=display_handle.display_id)
```



## day5总结

What can i do

- Describe transformers, tokens,context windows, API costs etc
- Contrast the leading 6 Frontier LLMs
- Confidently use the OpenAI API including streaming with markdown and JSON generation, and Ollama API

## week1作业

```python
# imports
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from openai import OpenAI
import ollama

# constans
MODEL_GPT = "gpt-4o-mini"
MODEL_LLAMA = "llama3.2"

# set up enviroment

load_dotenv()
openai = OpenAI()

# here is the question; type over this to ask something new

question = """
Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""

# prompts

system_prompt = "You are a helpful technical tutor who answers questions about python code, software engineering, data science and LLMs"
user_prompt = "Please give a detailed explanation to the following question: " + question

# messages
messages = [
    {"role" : "system","content" : system_prompt},
    {"role" : "user","content" : user_prompt}
]

# get chatGPT with stream

stream = openai.chat.completions.create(
	model = MODEL_GPT,
    messages = messages,
    stream = True
)

response = ""
display_handle = display(Markdown(""), display_id=True)
for chunk in stream:
    response += chunk.choices[0].delta.content or ''
    response = response.replace("```","").replace("markdown","")
    update_display(Markdown(response), display_id=display_handle.display_id)
    
# get llama
response = ollama.chat(
	model = MODEL_OLLAMA,
    messages = messages
)
reply = response["message"]["content"]
display(Markdown(reply))
```



# week2

## day1 start

```python
# connect to OpenAI, Anthropic and Google 
# All 3 APIs are similar

openai = OpenAI()

claude = anthropic.Anthropic()

google.generativeai.configure()
```

## Asking LLMs to tell a joke

It turns out that LLMs don't do a great job of telling jokes! Let's compare a few models. Later we will be putting LLMs to better use!

### What information is included in the API

Typically we'll pass to the API:

- The name of the model that should be used
- A system message that gives overall context for the role the LLM is playing
- A user message that provides the actual prompt

There are other parameters that can be used, including **temperature** which is typically between 0 and 1; higher for more random output; lower for more focused and deterministic.

```python
system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"
```

```python
prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]
```

```python
# GPT-3.5-Turbo

completion = openai.chat.completions.create(model = 'gpt-3.5-turbo', messages=prompts)
print(completion.choices[0].content)
```

```python
# GPT-4o-mini
# Temperature setting controls creativity

completion = openai.chat.completions.create(
	model='gpt-4o-mini',
    messages=prompts,
    temperature=0.7
)
print(completion.choices[0].content)
```

```python
# GPT-4o

completion = openai.chat.completions.create(
	model='gpt-4o',
    messages=prompts,
    temperature=0.4
)
print(completion.choices[0].content)
```

```python
# claude 3.5 Sonnet
# API needs system messages provided separately from user prompt
# Also adding max_tokens

message = claude.messages.create(
	model="claude-3-5-sonnet-20240620",
    max_tokens=200,
    temperature=0.7,
    system=system_messages,
    messages=[
        {"role": "user","content":user_prompt},
    ],
)

print(messages.content[0].text)
```

```python
# Claude 3.5 Sonnet again
# Now let's add in streaming back results

result = claude.messages.stream(
	model="claude-3-5-sonnet-20240620",
    max_tokens=200,
    temperature=0.7,
    system=system_messages,
    messages=[
        {"role","user":"content" : user_prompt},
    ],
)

with result as stream: // 典型的上下文管理器的用法
    for text in stream.text_stream:
        print(text, end="", flush=True) // 设置 end =""表示不添加换行符，而是连续打印内容。
        // flush = True 表示会强制立即将缓冲区的内容输出到控制台。
```

```python
# The API for Gemini has a slightly different structure.
# I've heard that on some PCs, this Gemini code causes the Kernel to crash.
# If that happpents to you, please skip this

gemini = google.generativeai.GenerativeModel(
	model_name='gemini-1.5-flash',
    system_instruction=system_message
)
response = gemini.generate_content(user_prompt)
print(response.text)
```

```python
# As an alternative way to use Gemini that bypasses Google's python API libraty,
# Google has recently released new endpoints that means you can use Gemini via the client libraries for OpenAI!

gemini_via_openai_client = OpenAI(
	api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = gemini_via_openai_client.chat.completions.create(
	model="gemini-1.5-flash",
    messages=prompts
)
print(response.choices[0].message.content)
```

```python
# Have it stream back results in markdown

stream = openai.chat.completions.create(
	model='gpt-4o',
    messages=prompts,
    temperature=0.7,
    stream=True
)

reply =""
display_handle = display(Markdown(""), display_id=True)
for chunk in stream:
    reply += chunk.choices[0].delta.content or ''
    reply = reply.replace("```","").replace("markdown","")
    update_display(Markdown(reply), display_id=display_handle.display_id)
```

## And now for some fun - an adversarial between Chatbots..

You're already familar with prompts being organized into lists like:

```python
[ 
    {"role": "system","content"： "system message here"},
    {"role": "user", "content": "user prompt here"}
]
```

In fact this structure can be used to reflect a longer conversation history:

```python
[
    {"role": "system", "content": "system message here"},
    {"role": "user", "content": "first user prompt here"},
    {"role": "assistant", "content": "the assistant's response"},
    {"role": "user", "content": "the new user prompt"},
]
```

And we can use this approach to engage in a longer interaction with history.

```python
# Let's make a conversation between GPT-4o-mini and Claude-3-haiku
# We're using cheap versions of modles so the costs will be minimal

gpt_model = "gpt-4o-mini"
claude_modle = "claude-3-haiku-20240307"

gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

claude_system = "You are a very polite, courteous chatbot. You try to agree with \ everything the other person says, or find common ground. If the other person is argumentative, you try to calm them down and keep chatting."

gpt_messages = ["Hi there"]
claude_messages = ["Hi"]
```

```python
def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, claude in zip(gpt_messages, claude_messages):
        messages.append({"role": "assistant", "content": "gpt"})
        messages.append({"role": "user", "content": claude})	completion = openai.chat.completions.create(
        model = gpt_model,
        messages = messages)
        return completion.choices[0].content
```

zip函数

- 作用：
  - zip 是 Python内置函数，用于将多个可迭代对象（如列表，元组等）“并行”打包成元组。
  - 每次从多个迭代对象中分别去除一个元素，组合成一个元素，形成一个新的可迭代对象。
  - 如果多个可迭代对象的长度不同，则会以最短的那个为准，忽略多余的元素。

```python
def call_claude():
    messages = []
    for gpt, claude_messages in zip(gpt_messages, claude_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "assistant", "content": claude_message})													messages.append({"role": "user", "content: gpt_messages[-1]"})												messgae = claude.messages.create(
        model = claude_model,
        system=claude_system,
        messages=messages,
        max_tokens = 500)										return messages.content[0].text
```

```python
gpt_messages = ["Hi there"]
claude_messages = ["Hi"]

print(f"GPT:\n{gpt_messages[0]}\n")
print(f"Claude:\n{claude_messages[0]}\n")

for i in range(5):
    gpt_next = call_gpt()
    print(f"GPT:\n{gpt_next}\n")
    gpt_messages.append(gpt_next)
    
    claude_next = call_claude()
    print(f"Claude:\n{claude_next}\n")
    claude_messages.append(claude_next)
```

## day1 总结

What can i already do

- Describe transformers and the leading 6 Frontier LLMs
- Confidently use the OpenAI API including streaming with markdown and JSON generation
- Use the Anthropic and Google APIs

## day2 start

## day2 总结

what can i already do

- Confidently use the OpenAI API including streaming with markdown and JSON generation
- Use the Anthropic and Google APIs
- Build UIs for your solutions

## day3 start

## Conversational AI - aka Chatbot!

```python
# imports

import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
```

```python
# Load enviroment variables in a life called .env
# Print the key prefixes to help with any debugging

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
if anthropic_api_key:
    print(f"Anthropic Api Key exists and begins {anthropic_api_key[:7]}")
else:
    print(f"Anthropic Api Key not Set")
if google_api_key:
    print(f"Google Api Key exists and begins {google_api_key[:8]}")
else:
    print("Google Api Key not set")
```

```python
# Initialize

openai = OpenAI()
MODEL = 'gpt-4o-mini'
```

```python
system_message = "You are a helpful assistant"
```

### Reminder of the structure of prompt messages to OpenAI:

```python
[
    {"role": "system","content": "system message here"},
    {"role": "user", "content": "first user prompt here"},
    {"role": "assistant", "content": "the assistant's response"},
    {"role": "user", "content": "the new user prompt"},
]

//We will write a function ``chat(message, history)`` //where:messgae is the prompt to use history is a list of pairs of user message with assistant's reply
[
    ["user said this", "assistant replied"],
    ["then user said this", "and assistant replied again"],
    ...
]
```

```python
def chat(message, history):
    messagess = [{"role":"system", "content" :system_message}]
    for user_message, assistant in  history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role" :"user", "content":message})
    
    print("History is:")
    print(history)
    print("And messages is:")
    print(messages)
    
    stream = openai.chat.completions.create(model = MODEL, messages = messages, stream=True)
    response =""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response
```

And then enter Gradio's magic

```python
gr.ChatInterface(fn=chat, type="messages").launch()
```

Other fun

```python
system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales evemt.'\
Encourage the customer to buy hats if they are unsure what to get."

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    
    stream = openai.chat.completions.create(model=MODEL,messages=messages, stream=True)
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response
```

```python
gr.ChatInterface(fn=chat, type="messages").launch()
```



## day3 总结

what can i already do

- Describe transformers and explain key terminolgy
- Confidently code with the APIs for GPT, Claude and Gemini
- Build an AI Chatbot Assistant including an interactive UI

## day 4 start

## Tools

Allows Frontier models to connect with external functions

- Richer(更富有的) responses by extending knowledge
- Ability to carry out actions within the application
- Enhanced(增强) capabilities (能力)，like calculations(计算机)

How it works

- In a request to the LLM, specify available Tools
- The reply(答复) is either(要么是) Text,or a request to run a Tool
- We run the Tool and call the LLM with the results

**Common Use Cases For Tools**

Function Calls can enable assistants to：

- Fetch data or add
- Take action like booking
- Perform calculations
- Modify the UI

构建一个知情的航空公司客户支持代理

```python
# import

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
```

```python
# Initiallization

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key :
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
MODEL = "gpt-4o-mini"
openai = OpenAI()

# As an alternative, if you'd like to use Ollama instead of OpenAI
# Check that Ollama is running for you locally 
MODEL = "llama3.2"
openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
```

```python
system_message = "You ar a helpful assistant for an Airline called FlightAI."
system_message += "Give short, courteous answers, no more than 1 sentence."
system_message += "Always be accurate. If you don't konw the answer, say so."
```

```python
def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role" : "user", "content" : message}]
    response = openai.chat.completions.create(modle=MODEL, messages=messages)
    return response.choices[0].message.content
gr.ChatInterface(fn=chat, type="messages").launch()
```

**Tools**

Tools are an incredibly powerful feature provided by the frontier LLMs.

With tools, you can write a function, and have the LLM call that function as part of its response.

Sounds almost spooky.. we're giving it the power to run code on our machine?

```python
# Let's start by making a useful function

ticket_prices = {"london": "$799", "paris": "$899","tokyo": "$1400", "berlin": "$499"}

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}“)
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")
```

```python
# There's a particular dictionary structure that's required to describe our function:

price_funtion = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to konw the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}
```

```python
# And this is included in a list of tools:

tools = [{"type": "function", "function": price_function}]
```

**Getting OpenAI to use our Tool**

There's some fiddly stuff to allow OpenAI "to call our tool"

What we actually do is give the LLM the opportunity to inform us that it wants us to run the tool.

Here's how the new chat function looks:

```python
def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + hsitory + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    
    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
        
        return response.choices[0].message.content
```

```python
# We have to write that function handle_tool_call:

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city, "price": price}),
        "tool_call_id" : tool_call.id
    }
    return response, city
```

```
gr.ChatInterface(fn= chat, type = "messages").launch()
```

## day4 总结

What can i do now

- Describe transformers and explain key terminology
- Confidently code with the APIs for GPT, Claude and Gemini
- Build an Ai Assistant using Tools for enhanced expertise

## day5 start

DEFINING AGENTS(代理)

**Software(软件) entities(实体) that can autonomously(自主的) perform tasks**



Common characteristics(特质) of Agent

- Autonomous
- Goal-oriented
- Task specific

Designed(有计划的) to work as part of an Agent Framework to solve(解决) complex(复杂的) problems with limited human invelvement

- Memory/persistence(持久层)
- Decision-making/orchestration
- Planning capabilities
- Use of tools;potentially(可能的) connecting to databases or the internet

**What we are about to do**

- **Image Generation**
  - Use the OpenAI interface to generate images
- **Make Agents**
  - Create Agents generate sound and images
- **Make an Agent Framework**
  - Teach our AI Assistant to speak and draw

**Project - Airline AI Assistant**

We'll now bring together what we've learned to make an AI Customer(顾客) Support  assistant(助手) for an Airline

```python
# imports 

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
```

```python
# Initialization

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
MODEL = "gpt-4o-mini"
openai = OpenAI()
```

```python
system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."
```

```python
def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content
gr.ChatInterface(fn=chat, type="messages").launch()
```

## Tools

Tools are an incredibly powerful feature provided by the frontier LLMs.

With tools, you can write a function, and have the LLM call that function as part of its response.

Sounds almost spooky.. we're giving it the power to run code on our machine?

Well, kinda.

```python
# Let's start by making a useful function

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")
```

```python
# There's a particular dicitionary structure that's required to describe our function:

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "destination_city": {
            "type": "string",
            "description": "The city that the customer wants to travel to",
        },
    },
    "required": ["destination_city"],
    "additionalProperties": False
}
```

```python
# And this is included in a list of tools:

tools = [{"type": "function": price_function}]
```

**Getting OpenAI to use our Tool**

There's some fiddly stuff to allow OpenAI "to call our tool"

What we actually do is give the LLM the opportunity to inform us that it wants us to run the tool.

Here's how the new chat function looks:

```python
def chat(message, history):
    messgae = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    
    if response.choices[0].finish_reason="tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
     
    return respnse.choices[0].message.content
```

```python
# We have to write that function handle_tool_call:

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city,"price": price}),
        "tool_call_id": tool_call.id
    }
    return response, city
```

```python
gr.ChatInterface(fn=chat, type="messages").launch()
```

接下来去构建绘画模型

```python
# Some imports for handling images

import base64
from io import BytesIO
from PIL import Image
```

```python
def artist(city):
    image_response = openai.images.generate(
    	model="dall-e-3",
        prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
        size="1024*1024",
        n=1,
        response_format="b64_json",
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))
```

**Audio**

And let's make a function talker that uses OpenAI's speech model to generate Audio

Trouleshooting(故障排除) Audio issues

Variation 1

```python
import base64
from io import BytesIO
from PIL import Image
from IPython.display import Audio, display

def talker(message):
    response = openai.audio.speech.create(
    	model="tts-1",
        voice="onyx",
        input=message
    )
    audio_stream = BytesIO(response.content)
    output_filename = "output_audio.mp3"
    with open(output_filename, "wb") as f:
        f.write(audio_stream.read())
        
    # Play the generated audio
    display(Audio(output_filename, autoplay=True))
    
talker("Well, hi there")
```

Variation 2

```python
import tempfile
import subprocess
from io import BytesIO
from pydub import AudioSegment
import time

def play_audio(audio_segment):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_audio.wav")
    try:
        audio_segment.export(temp_path, format="wav")
        subprocess.call([
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-hide_banner",
            temp_path
        ], studot=suprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        try:
            os.remove(temp_path)
        excpt Exception:
            pass
        
def talker(message):
    response = openai.audio.speech.create(
    	model="tts-1",
        voice="onyx", # Also, try replacing onyx with alloy
        input=message
    )
    audio_stream = BytesIO(response.content)
    audio = AudioSement.from_file(audio_stream, format="mp3")
    play_audio(audio)
    
talker("Well hi there")
```

Variation 3

Let's try a completely different sound library

First run the next cell to install a new library, then try the cell below it.

`!pip install simpleaudio`

```python
from pydub import AudioSegment
from io import BytesIO
import tempfile
import os
import simpleaudio as sa

def talker(message):
    response = openai.audio.speech.create(
    	model="tts-1",
        voice="onyx", #Also, try replacing onyx with alloy
        input=message
    )
    
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    
    # Create a temporary file in a folder where you have write permissions
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir = os.path.expanduser("~/Documents")) as temp_audio_file:
        temp_file_name = temp_audio_file.name
        audio.export(temp_file_name, format="wav")
        
    # Load and play audio using simpleaudio
    wave_obj = sa.WaveObject.from_wave_file(temp_file_name)
    play_obj = wave_obj.play()
    play_obj.wait_done() # Wait for playback to finish
    
    # Clean up the temporary file afterward
    os.remove(temp_file_name)
talker("Well hi there")
```

# Our Agent Framework

The term 'Agentic AI' and Agentization is an umbrella term that refers to a number of techniques, such as:

1. Breaking a complex problem into smaller steps, with multiple LLMs carrying out specialized tasks
2. The ability for LLMs to use Tools to give them additional capabilities
3. The 'Agent Environment' which allows Agents to collaborate
4. An LLM can act as the Planner, dividing bigger tasks into smaller ones for the specialists
5. The concept of an Agent having autonomy / agency, beyond just responding to a prompt - such as Memory

We're showing 1 and 2 here, and to a lesser extent 3 and 5. In week 8 we will do the lot!

```python
def chat(history):
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    image = None
    
    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city= handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        imapge = artist(city)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
        
    reply = resopnse.choices[0].message.content
    history += [{"role":"assistant", "content":reply}]
    
    # Comment out or delete the next line if you'd rather skip Audio for now..
    talker(reply)
    
    return history, image
```

```python
# More involved Gradio code as we're not using the preset Chat interface!
# Passing in inbrowser=True in the last line will cause a Gradio window to pop up immediately.

wit gr.Blocks() as ui:
    with gr.Row():
        chatbot= gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row():
        clear = gr.Button("Clear")
        
    def do_entry(message, history):
        history += [{"role":"user", "content":message}]
        return "", history
    
    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
    	chat, inputs=chatbot, outputs=[chatbot, image_output]
    )
    clear.click(lamba: None, inputs=None, outputs=chatbot, queue=False)

ui.launch(inbrowser=True)
```



## day 5 总结

What can i now do

- Describe transformers and explain key terminology
- Confidently code with the APIs for GPT, Claude and Gemini
- Build a multi-modal AI Assistant with UI, Tools, Agents



# week3