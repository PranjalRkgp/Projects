## AdaptiQuiz: AI-Powered Adaptive Quiz Generator
An intelligent quiz platform that uses the Llama3 LLM to generate personalized and adaptive technical assessments based on your unique career profile and learning goals.

## Live Application
Access the live Streamlit app here: https://adaptiquiz.streamlit.app/
### Key Features
Deep Personalization: Configure your quiz based on your target job role, years of experience, tech stack, and desired companies.

Dynamic Question Generation: Utilizes Llama3 via the Groq API to create unique, high-quality questions on the fly. No static question banks!

Adaptive Difficulty: The quiz automatically increases in difficulty when you answer correctly, keeping you challenged.

Real-World Simulation: Mimics various interview formats like online tests, system design rounds, and take-home assignments.

Timed Environment: Each question is timed to simulate real test pressure.

Instant Feedback: Receive the correct answer and a detailed explanation immediately after each question.

## How It Works
The application follows a simple, user-driven process:

Configure: You fill out a detailed form specifying the test structure, your career profile, and the real-world scenario you want to simulate.

Generate: Based on your inputs, the backend sends a carefully crafted prompt to the Llama3 LLM to generate a relevant multiple-choice question.

Answer: You answer the question within the allotted time.

Evaluate & Adapt: Your answer is evaluated instantly. If you're correct, the difficulty for the next question may increase.

Learn: You receive a detailed explanation for the correct answer, helping you learn on the spot.

Repeat: The cycle continues until the quiz is complete, after which you get a final score and a summary of the concepts covered.

## Tech Stack
Framework: Streamlit

LLM Integration: LangChain

LLM API: Groq (for high-speed Llama3 inference)

Language: Python
