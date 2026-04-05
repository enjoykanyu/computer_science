# hello-agent第二章学习


```python
import re
import random

# 定义规则库:模式(正则表达式) -> 响应模板列表
rules = {
    r'I need (.*)': [
        "Why do you need {0}?",
        "Would it really help you to get {0}?",
        "Are you sure you need {0}?"
    ],
    r'Why don\'t you (.*)\?': [
        "Do you really think I don't {0}?",
        "Perhaps eventually I will {0}.",
        "Do you really want me to {0}?"
    ],
    r'Why can\'t I (.*)\?': [
        "Do you think you should be able to {0}?",
        "If you could {0}, what would you do?",
        "I don't know -- why can't you {0}?"
    ],
    r'I am (.*)': [
        "Did you come to me because you are {0}?",
        "How long have you been {0}?",
        "How do you feel about being {0}?"
    ],
    r'.* mother .*': [
        "Tell me more about your mother.",
        "What was your relationship with your mother like?",
        "How do you feel about your mother?"
    ],
    r'.* father .*': [
        "Tell me more about your father.",
        "How did your father make you feel?",
        "What has your father taught you?"
    ],
    r'.*': [
        "Please tell me more.",
        "Let's change focus a bit... Tell me about your family.",
        "Can you elaborate on that?"
    ]
}

# 定义代词转换规则
pronoun_swap = {
    "i": "you", "you": "i", "me": "you", "my": "your",
    "am": "are", "are": "am", "was": "were", "i'd": "you would",
    "i've": "you have", "i'll": "you will", "yours": "mine",
    "mine": "yours"
}

def swap_pronouns(phrase):
    """
    对输入短语中的代词进行第一/第二人称转换
    """
    words = phrase.lower().split()
    swapped_words = [pronoun_swap.get(word, word) for word in words]
    return " ".join(swapped_words)

def respond(user_input):
    """
    根据规则库生成响应
    """
    for pattern, responses in rules.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            # 捕获匹配到的部分
            captured_group = match.group(1) if match.groups() else ''
            # 进行代词转换
            swapped_group = swap_pronouns(captured_group)
            # 从模板中随机选择一个并格式化
            response = random.choice(responses).format(swapped_group)
            return response
    # 如果没有匹配任何特定规则，使用最后的通配符规则
    return random.choice(rules[r'.*'])

# 主聊天循环
if __name__ == '__main__':
    print("Therapist: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Therapist: Goodbye. It was nice talking to you.")
            break
        response = respond(user_input)
        print(f"Therapist: {response}")


```



返回

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775361823245-04bf0a84-2bb0-4c80-9fdf-4a22fdf97aae.png" width="628" title="" crop="0,0,1,1" id="u00ef4667" class="ne-image">



##### 习题1
<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775374910055-2af1bef7-0419-4b67-841d-56b0a6b34bc4.png" width="512" title="" crop="0,0,1,1" id="ua8778442" class="ne-image">

回答（来源于大模型）

含义

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775375083276-fe276ba0-59d3-4ce6-8c69-b8a7269f93b5.png" width="403" title="" crop="0,0,1,1" id="u0af8fdc0" class="ne-image">





挑战1

常识问题对于AI来说有的例外情况，符号注意推论错误

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775375688990-26eb6211-1529-4c0e-8ef1-521eb67c5a59.png" width="810" title="" crop="0,0,1,1" id="u217f7b29" class="ne-image">



挑战2

框架可以存在对话历史，但不知道存在的、不变的的既定事实

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775375774083-7bf73af9-cdbe-4fa0-83c9-70522d17aa29.png" width="819" title="" crop="0,0,1,1" id="u243993bd" class="ne-image">



挑战3

组合爆炸，排列组合实际场景很复杂，全部列举出来叉乘储量情况相当多

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775375906888-9565cf93-eb44-4068-8c69-f1ebe1ac6269.png" width="527" title="" crop="0,0,1,1" id="uc26b5766" class="ne-image">



挑战4



<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775375997846-bfb2bdc4-16a6-423b-ac81-399fbebe7a5d.png" width="806" title="" crop="0,0,1,1" id="u5afc536c" class="ne-image">



挑战5

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775376052963-0fe58f7c-5b59-4c8f-b8b5-d742f6219056.png" width="449" title="" crop="0,0,1,1" id="u22f6008f" class="ne-image">



挑战6

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775376106951-feec6a11-0b1a-4f05-9f4e-3a8366b37c9c.png" width="564" title="" crop="0,0,1,1" id="u7b64b035" class="ne-image">





大模型智能体是否符合呢

符合的

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775377203917-958b1ec9-edfd-4636-91e8-77074949cf91.png" width="470" title="" crop="0,0,1,1" id="u79b4064f" class="ne-image">

不符合

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775377222939-94e5166a-831a-40eb-b5bd-b212b3df6c89.png" width="413" title="" crop="0,0,1,1" id="u7caf31de" class="ne-image">



<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775377236383-db91e419-b40d-438e-ae36-b343880e546a.png" width="404" title="" crop="0,0,1,1" id="ub28d22e9" class="ne-image">
