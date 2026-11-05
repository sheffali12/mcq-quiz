import random
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI(title="MCQ Quiz API")
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5)

LETTERS = ["A", "B", "C", "D"]

# LLM Structure 
class MCQ(BaseModel):
    question: str = Field(description="The question text")
    options: List[str] = Field(
        min_length=4, max_length=4,
        description="Exactly 4 answer options, without letter labels like 'A)'"
    )
    correct_index: int = Field(
        ge=0, le=3,
        description="Index (0-3) of the correct option in the options list"
    )
    explanation: str = Field(description="One or two sentences explaining why the answer is correct")

class MCQSet(BaseModel):
    items: List[MCQ]

# Prompt
quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert teacher who writes clear multiple choice questions."),
    ("human", "Generate {num_questions} {difficulty} level multiple choice questions about: {topic}. "
              "Each question must have exactly 4 options with exactly one correct answer. "
              "Make the wrong options plausible but clearly incorrect.")
])
quiz_chain = quiz_prompt | llm.with_structured_output(MCQSet)


class StoredQuestion(BaseModel):
    question: str
    options: List[str]
    correct_letter: str
    explanation: str

current_quiz: List[StoredQuestion] = []

# MODEls 
class QuizRequest(BaseModel):
    topic: str = Field(min_length=1)
    num_questions: int = Field(ge=1, le=20)
    difficulty: Literal["easy", "medium", "hard"]

class QuestionOut(BaseModel):
    number: int
    question: str
    options: dict[str, str] 

class QuizResponse(BaseModel):
    questions: List[QuestionOut]

class AnswerRequest(BaseModel):
    option: str = Field(pattern="^[A-Da-d]$", description="Your chosen option: A, B, C or D")

class AnswerResponse(BaseModel):
    is_correct: bool
    your_option: str
    # Wrong 
    correct_option: Optional[str] = None   
    correct_answer: Optional[str] = None   
    explanation: str

# first point 
@app.post("/quiz", response_model=QuizResponse)
def create_quiz(req: QuizRequest):
    try:
        mcqs = quiz_chain.invoke(req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    stored: List[StoredQuestion] = []
    for m in mcqs.items:
        correct_text = m.options[m.correct_index]
        shuffled = m.options[:]
        random.shuffle(shuffled)  # LLMs tend to favour certain positions, so shuffle
        stored.append(StoredQuestion(
            question=m.question,
            options=shuffled,
            correct_letter=LETTERS[shuffled.index(correct_text)],
            explanation=m.explanation,
        ))

    current_quiz.clear()         
    current_quiz.extend(stored)

    return QuizResponse(
        questions=[
            QuestionOut(
                number=i,
                question=q.question,
                options=dict(zip(LETTERS, q.options)),
            )
            for i, q in enumerate(stored, 1)
        ],
    )

@app.post("/quiz/{number}/answer", response_model=AnswerResponse)
def check_answer(number: int, req: AnswerRequest):
    if not current_quiz:
        raise HTTPException(status_code=404, detail="No quiz yet. Create one with POST /quiz first.")
    if not 1 <= number <= len(current_quiz):
        raise HTTPException(status_code=404, detail="Question number out of range")

    q = current_quiz[number - 1]
    chosen = req.option.upper()
    is_correct = chosen == q.correct_letter

    return AnswerResponse(
        is_correct=is_correct,
        your_option=chosen,
        correct_option=None if is_correct else q.correct_letter,
        correct_answer=None if is_correct else q.options[LETTERS.index(q.correct_letter)],
        explanation=q.explanation,
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Ques:app", host="127.0.0.1", port=8085, reload=True)