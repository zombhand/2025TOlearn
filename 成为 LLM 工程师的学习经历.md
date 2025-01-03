# 成为 LLM 工程师的学习经历

# week1

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
elif not api_key.startswith("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"):
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

## day1总结

what can i do ALREADY

- Use Ollama to run LLMs locally on your box
- Write code to call OpenAl's frontier models
- Distinguish between the System and User prompt
- Summarization - applicable to many commercial problems

# week2

# week3