"""
Dataset Generator for AI vs Human Text Detection
Uses realistic samples representing common writing styles
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ── Human-written samples ──────────────────────────────────────────────────────
human_texts = [
    # Casual / personal
    "I've been thinking about switching careers for a while now. It's scary honestly, but staying in a job that drains you every single day isn't sustainable. I talked to my friend about it and she said just go for it, life's too short.",
    "okay so i finally tried that new ramen place everyone keeps talking about and honestly? overrated. the broth was fine but the noodles were too soft and it was way too expensive for the portion size",
    "My dad called me out of nowhere today. We haven't spoken in months. Didn't know what to say honestly, just kind of listened. It's complicated.",
    "Been trying to get into running for like the third time in my life lol. Did 2km today without stopping which is pathetic but also kind of proud of myself",
    "The meeting ran over AGAIN. I had back to back calls from 9am and didn't eat until 3. This is not a sustainable way to work.",
    "finally finished reading that book my sister recommended. took me four months because i kept picking it up and putting it down. worth it though, the ending was not what i expected at all",
    "Can't sleep. Brain won't shut off. Made a list of everything I'm worried about and somehow that made it worse",
    "we had the worst group project experience in college. one person did nothing, one person rewrote everything the night before, and somehow we still got a B+",

    # Informal academic / student writing
    "In my opinion climate change is one of the biggest problems we face today. Scientists have been warning us for decades but governments keep ignoring it because of economic interests. Something needs to change and fast.",
    "I think the main reason social media is bad for teenagers is that it creates unrealistic expectations. You're constantly comparing yourself to people who only post their highlights. It messes with your self esteem.",
    "For my dissertation I looked at how remote work affects productivity. I expected clear results but the data was really mixed. Some people thrived, others really struggled, and it seemed to depend a lot on home environment.",
    "The thing about machine learning that nobody tells you upfront is that most of your time is spent cleaning data, not building models. Like 80% of it is just dealing with missing values and weird formatting issues.",
    "I tried to replicate the results from the paper but kept getting different numbers. Eventually figured out they'd normalised the data differently than they described in the methods section. Super frustrating.",

    # Professional but personal voice
    "After three years at the same company I finally handed in my notice. It was the right call but I didn't sleep the night before. Change is hard even when you know it's necessary.",
    "We launched the new feature last Tuesday and the user feedback has been really encouraging. A few edge cases we didn't anticipate but nothing critical. The team worked incredibly hard on this one.",
    "I've been mentoring a junior developer for the past few months and honestly I think I'm learning as much from her as she is from me. Explaining things clearly forces you to actually understand them.",
    "The code review took longer than expected because we found an architectural issue that's been sitting there for two years. Better to fix it now than let it compound.",
    "Tried a new approach to the project planning this sprint — gave people more autonomy over how they hit the goals rather than dictating process. Results were noticeably better.",

    # Opinions and analysis, informal
    "I don't think AI will replace programmers, but it will definitely change what programmers spend their time on. The boring repetitive stuff will be automated. The hard thinking won't be.",
    "people keep saying remote work is the future but i think most companies are going to quietly force everyone back to the office within five years. there's too much middle management that needs to justify itself",
    "The problem with most productivity advice is it's written by people who already have ideal conditions — quiet office, no kids, financial stability. Doesn't translate for most people.",
    "Hot take: most meetings could be emails and most emails could be a Slack message and most Slack messages don't need to exist at all",
    "I find it genuinely hard to trust AI-generated text now. Even when it's accurate it feels weirdly smooth, like there are no rough edges. Real writing has rough edges.",

    # Errors, corrections, uncertainty
    "not 100% sure about this but i think the deadline is friday? someone should double check with the manager",
    "I might be wrong here but I'm pretty sure that's not how TCP/IP actually works. The handshake happens before any data is transferred, not after.",
    "tbh i don't fully understand how transformers work under the hood. like i can use them and get results but the attention mechanism still feels a bit magical to me",
    "we probably should have tested this on a larger sample size. the results look promising but with n=47 i wouldn't want to draw strong conclusions",
]

# ── AI-generated samples ───────────────────────────────────────────────────────
ai_texts = [
    # Typical LLM explanatory style
    "Large language models (LLMs) represent a significant advancement in artificial intelligence, enabling machines to generate coherent and contextually relevant text across a wide range of domains. These models are trained on vast corpora of text data, allowing them to learn complex patterns in language.",
    "There are several key factors to consider when evaluating the performance of a machine learning model. First, it is important to select appropriate evaluation metrics that align with the specific goals of the task. Second, the model should be tested on a held-out test set to ensure generalisability.",
    "Climate change poses one of the most significant challenges facing humanity today. Rising global temperatures, driven by increased concentrations of greenhouse gases in the atmosphere, are leading to more frequent and severe weather events, rising sea levels, and disruptions to ecosystems worldwide.",
    "Remote work has become increasingly prevalent in recent years, particularly following the global pandemic. Research suggests that remote work can offer numerous benefits, including increased flexibility, reduced commute times, and improved work-life balance for many employees.",
    "To summarise, the implementation of agile methodologies in software development teams has been shown to improve productivity, enhance collaboration, and increase responsiveness to changing requirements. However, successful adoption requires strong organisational commitment and cultural change.",

    # LLM list-making tendency
    "There are several important steps to follow when setting up a Python development environment. First, install Python from the official website. Second, create a virtual environment using venv or conda. Third, install the required dependencies using pip. Finally, configure your IDE of choice.",
    "When writing a personal statement, it is essential to: (1) clearly articulate your motivations and goals, (2) highlight relevant experience and skills, (3) demonstrate knowledge of the programme or role, and (4) conclude with a forward-looking statement about your aspirations.",
    "The benefits of regular exercise are well-documented and include: improved cardiovascular health, enhanced mental well-being, increased muscle strength and flexibility, better sleep quality, and reduced risk of chronic diseases such as diabetes and obesity.",

    # Overly balanced / hedged LLM opinions
    "The question of whether artificial intelligence will replace human workers is complex and multifaceted. While AI has the potential to automate certain repetitive and routine tasks, it is also creating new categories of employment and augmenting human capabilities in meaningful ways.",
    "Social media has both positive and negative effects on mental health. On one hand, it enables connection, community building, and access to information. On the other hand, excessive use has been associated with increased anxiety, depression, and social comparison.",
    "It is worth noting that while large language models have demonstrated impressive capabilities across a range of natural language processing tasks, they also exhibit significant limitations, including a tendency to hallucinate facts, difficulty with logical reasoning, and lack of true understanding.",

    # Formal academic LLM style
    "This study investigates the relationship between sleep duration and cognitive performance in university students. A total of 120 participants completed a series of standardised cognitive assessments following periods of restricted and unrestricted sleep. Results indicate a statistically significant correlation between sleep duration and performance on tasks requiring sustained attention.",
    "The transformer architecture, introduced by Vaswani et al. in 2017, has fundamentally transformed the field of natural language processing. By replacing recurrent neural networks with self-attention mechanisms, transformers are able to process sequences in parallel, resulting in significant improvements in training efficiency and model performance.",
    "Neural network interpretability has emerged as a critical area of research as deep learning models are increasingly deployed in high-stakes domains. Understanding the internal representations learned by these models is essential for ensuring their reliability, fairness, and alignment with human values.",

    # Smooth, slightly generic narrative
    "Throughout my academic and professional journey, I have developed a deep passion for the intersection of technology and human experience. My background in software engineering has provided me with a strong technical foundation, while my research experience has cultivated my analytical and critical thinking skills.",
    "I am excited to apply for this opportunity as it aligns perfectly with my academic background, professional experience, and long-term career aspirations. I believe that my unique combination of skills and experiences makes me an ideal candidate for this position.",
    "In conclusion, the rapid advancement of artificial intelligence technologies presents both extraordinary opportunities and significant challenges for society. It is essential that researchers, policymakers, and industry leaders work collaboratively to ensure that these technologies are developed and deployed in a responsible and ethical manner.",

    # Common LLM patterns in responses
    "Certainly! Here is a comprehensive overview of the key concepts in machine learning. Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve their performance over time without being explicitly programmed.",
    "Great question! The main difference between supervised and unsupervised learning lies in the nature of the training data. In supervised learning, the model is trained on labelled data, meaning each input is paired with a corresponding output. In unsupervised learning, the model works with unlabelled data.",
    "I hope this explanation has been helpful. To recap the main points: transformers use self-attention mechanisms to process sequential data, they can be pre-trained on large corpora and fine-tuned for specific tasks, and they have achieved state-of-the-art performance across numerous NLP benchmarks.",

    # Professional but suspiciously polished
    "The implementation was completed successfully and all unit tests passed. The code has been refactored to improve readability and maintainability, following best practices for clean code. Documentation has been updated to reflect the changes made during this sprint.",
    "After careful consideration of the available options, we recommend proceeding with the proposed solution. This approach offers the best balance of performance, scalability, and maintainability, while aligning with the organisation's long-term technical strategy.",
    "It is important to approach this topic with nuance and sensitivity, acknowledging the diverse perspectives and lived experiences of those affected. A one-size-fits-all solution is unlikely to be effective, and any intervention should be tailored to the specific context and needs of the community.",
    "The dataset used in this analysis consists of 10,000 samples collected over a period of six months. Data preprocessing involved removing duplicate entries, handling missing values through mean imputation, and normalising continuous features to a range of 0 to 1.",
    "In the context of natural language processing, tokenisation refers to the process of breaking down a piece of text into smaller units called tokens. These tokens can represent words, subwords, or characters, depending on the tokenisation strategy employed.",
]

# Build DataFrame
records = []
for text in human_texts:
    records.append({"text": text.strip(), "label": 0, "label_name": "human"})
for text in ai_texts:
    records.append({"text": text.strip(), "label": 1, "label_name": "ai_generated"})

df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/dataset.csv", index=False)
print(f"Dataset created: {len(df)} samples")
print(f"  Human: {(df.label==0).sum()}")
print(f"  AI-generated: {(df.label==1).sum()}")
print(df.head(3)[["label_name","text"]].to_string())
