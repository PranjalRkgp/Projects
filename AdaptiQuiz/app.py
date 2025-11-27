import streamlit as st 
import time
import random
from datetime import timedelta
from typing import List

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# --- 1. CONFIGURATION & SETUP ---

# Page Config
st.set_page_config(page_title="AI Skill Assessment", page_icon="🧪", layout="wide")

# Initialize LLM (Ensure you have .streamlit/secrets.toml set up)
try:
    llm = ChatGroq(
        groq_api_key=st.secrets["groq"]["api_key"], 
        model_name="llama3-70b-8192",
        temperature=0.7
    )
except Exception as e:
    st.error("Error initializing Groq. Please check your .streamlit/secrets.toml file.")
    st.stop()

# --- 2. DATA MODELS (CV Point: JSON-Schema / Automated Validation) ---

class QuizQuestion(BaseModel):
    """
    Strict Schema for the LLM response. 
    This enforces the structure and enables 'Chain of Verification' logic.
    """
    question: str = Field(description="The question text")
    choices: List[str] = Field(description="List of exactly 4 distinct choices")
    correct_answer: str = Field(description="The correct choice text, must match one choice exactly")
    explanation: str = Field(description="Detailed explanation of why the answer is correct")
    concept: str = Field(description="The specific technical concept being tested (1-3 words)")

# --- 3. SESSION STATE MANAGEMENT ---

def initialize_session_state():
    """Initialize or reset session state variables"""
    default_values = {
        'quiz_started': False,
        'user_inputs': None,
        'current_question': None,
        'question_count': 0,
        'score': 0,
        'answers': [],
        'show_answer': False,
        'time_per_question': 0,
        'asked_concepts': [],
        'current_difficulty': None,
        'current_question_style': None,
        'timer_expired': False,
        'answer_submitted': False,
        'force_rerun': False,
        'consecutive_correct': 0  # CV Point: For Adaptive Logic
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- 4. LOGIC FUNCTIONS ---

def calculate_time_per_question(total_time, total_questions):
    """Convert time string to seconds per question"""
    if "hour" in total_time:
        hours = int(total_time.split()[0])
        total_seconds = hours * 3600
    elif "mins" in total_time:
        minutes = int(total_time.split()[0])
        total_seconds = minutes * 60
    else:  # 90 mins case
        total_seconds = 90 * 60
    
    return total_seconds / total_questions

def format_time(seconds):
    """Format seconds into MM:SS format"""
    return str(timedelta(seconds=int(seconds)))[2:]

def update_difficulty(is_correct):
    """
    CV Point: Adaptive Evaluation & Dynamic Progression
    Logic: 
    - 2 Consecutive Correct -> Level Up
    - 1 Incorrect -> Reset Streak (and optionally level down)
    """
    if not st.session_state.user_inputs["Testing Structure"]["Adaptive Progression"]:
        return

    difficulty_levels = ["Beginner", "Intermediate", "Senior Level", "Hiring Challenge"]
    current_idx = difficulty_levels.index(st.session_state.current_difficulty)

    if is_correct:
        st.session_state.consecutive_correct += 1
        # Check threshold for level up
        if st.session_state.consecutive_correct >= 2 and current_idx < len(difficulty_levels) - 1:
            st.session_state.current_difficulty = difficulty_levels[current_idx + 1]
            st.session_state.consecutive_correct = 0 # Reset streak
            st.toast(f"🚀 Performing well! Difficulty increased to: {st.session_state.current_difficulty}")
    else:
        st.session_state.consecutive_correct = 0 # Streak broken
        # Optional: Punishing logic (Level down on failure)
        if current_idx > 0:
            st.session_state.current_difficulty = difficulty_levels[current_idx - 1]
            st.toast(f"📉 Adjusting pace. Difficulty set to: {st.session_state.current_difficulty}")

def get_question(user_input_dict, asked_concepts, current_difficulty, question_style):
    """
    CV Point: COVe, JSON-Schema Prompting, Contextual Deduplication, Safe API-Call
    """
    
    # 1. Setup Parser (Automated Validation)
    parser = JsonOutputParser(pydantic_object=QuizQuestion)
    
    # 2. Extract User Context
    profile = user_input_dict["User Career Profile"]
    tech_stack_str = ", ".join(profile.get("Tech Stack", ["General Coding"]))
    role = profile.get("Target Role", ["Developer"])[0]
    exp = profile.get("Years of Experience", 1)

    # 3. Define Prompt
    system_prompt = """
    You are an expert technical interviewer. 
    Generate a {difficulty} level multiple-choice question about {tech_stack} for a {role} role with {exp} years experience.
    
    Question Style: {style}
    
    CRITICAL RULES:
    1. Do NOT ask about these previously covered concepts: {asked_concepts}
    2. Ensure the 'correct_answer' matches one of the 'choices' exactly.
    3. Return ONLY valid JSON.
    
    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # 4. Create Chain
    chain = prompt | llm | parser

    # 5. Resilience Loop (CV Point: Safe API-call rate limit hit)
    max_retries = 3
    base_delay = 1.5 # seconds
    
    for attempt in range(max_retries):
        try:
            response = chain.invoke({
                "difficulty": current_difficulty,
                "tech_stack": tech_stack_str,
                "role": role,
                "exp": exp,
                "style": question_style,
                "asked_concepts": ", ".join(asked_concepts[-5:]), # Contextual Deduplication
                "format_instructions": parser.get_format_instructions()
            })
            return response

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Rate limit" in error_msg:
                # Exponential Backoff
                wait_time = base_delay * (2 ** attempt)
                st.toast(f"High traffic (Rate Limit). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Parsing/Validation Error - Immediate Retry
                print(f"Validation Error (Attempt {attempt+1}): {e}")
                continue
    
    # 6. Zero-Fallback Safety Net
    return {
        'question': "We are experiencing high API traffic. Please identify the concept of 'Rate Limiting'.",
        'choices': ["Controlling traffic flow", "Deleting database", "Increasing speed", "None"],
        'correct_answer': "Controlling traffic flow",
        'explanation': "This is a fallback question generated due to API congestion.",
        'concept': "API Reliability"
    }

# --- 5. UI COMPONENTS ---

def display_configuration_form():
    """Display the initial configuration form"""
    st.title("🧪 Adaptive Skill Assessment")
    st.caption("Powered by Llama-3, LangChain & Pydantic Validation")

    with st.form("config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("1. Structure")
            test_name = st.text_input("Test Name", value="Full Stack Evaluation")
            question_count = st.slider("Question Count", 1, 50, 5)
            time_limit = st.selectbox("Time Limit", ["30 mins", "1 hour", "90 mins"])
            adaptive_progression = st.toggle("Adaptive Difficulty", value=True, help="Increases difficulty after 2 consecutive correct answers.")

        with col2:
            st.header("2. Profile")
            target_role = st.multiselect("Role", ["Full Stack Developer", "Data Scientist", "SDE II"], default=["SDE II"])
            difficulty = st.selectbox("Start Difficulty", ["Beginner", "Intermediate", "Senior Level", "Hiring Challenge"])
            tech_stack = st.text_input("Tech Stack", value="Python, React, SQL, System Design")
            question_style = st.multiselect("Style", ["Scenario based", "Debugging", "Conceptual"], default=["Scenario based"])

        submitted = st.form_submit_button("Start Assessment", type="primary")

        if submitted:
            user_inputs = {
                "Testing Structure": {
                    "Test Name": test_name,
                    "Question Count": question_count,
                    "Time Limit": time_limit,
                    "Initial Difficulty": difficulty,
                    "Adaptive Progression": adaptive_progression,
                    "Question Styles": question_style
                },
                "User Career Profile": {
                    "Target Role": target_role,
                    "Years of Experience": 3, # Hardcoded for brevity, can make dynamic
                    "Tech Stack": [t.strip() for t in tech_stack.split(",")]
                }
            }
            
            st.session_state.user_inputs = user_inputs
            st.session_state.quiz_started = True
            st.session_state.time_per_question = calculate_time_per_question(time_limit, question_count)
            st.session_state.current_difficulty = difficulty
            st.rerun()

def display_question_with_timer():
    """Display the question with a real-time countdown timer"""
    question_data = st.session_state.current_question
    total_questions = st.session_state.user_inputs["Testing Structure"]["Question Count"]
    time_per_question = st.session_state.time_per_question
    
    # Progress Bar
    progress = (st.session_state.question_count) / total_questions
    st.progress(progress)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Question", f"{st.session_state.question_count + 1}/{total_questions}")
    c2.metric("Difficulty", st.session_state.current_difficulty)
    c3.metric("Score", st.session_state.score)
    timer_placeholder = c4.empty()
    
    st.divider()

    # Question Display
    st.subheader(question_data.get('question', 'Error loading question'))
    st.caption(f"Topic: {question_data.get('concept', 'General')} | Style: {st.session_state.current_question_style}")
    
    # Timer Logic
    start_time = time.time()
    end_time = start_time + time_per_question
    
    # Answer Form
    with st.form(key="answer_form"):
        choices = question_data.get('choices', [])
        user_choice = st.radio("Select your answer:", choices, index=None)
        submitted = st.form_submit_button("Submit Answer", type="primary")

    # Timer Loop (only runs if answer not submitted yet)
    if not submitted and not st.session_state.force_rerun:
        while True:
            remaining = end_time - time.time()
            if remaining <= 0:
                st.session_state.timer_expired = True
                st.session_state.show_answer = True
                # Record timeout
                st.session_state.answers.append({
                    "question": question_data['question'],
                    "user_answer": None,
                    "correct": False,
                    "concept": question_data.get('concept')
                })
                st.session_state.force_rerun = True
                break
            
            timer_placeholder.metric("Time Left", format_time(max(0, remaining)))
            time.sleep(0.1)
            
            # Check if user clicked submit in the UI (Streamlit handles this on rerun, 
            # but we need to break loop if state changes)
            if submitted: 
                break

    # Handle Submission
    if submitted and user_choice:
        st.session_state.answer_submitted = True
        
        correct_answer = question_data.get('correct_answer')
        is_correct = (user_choice == correct_answer)
        
        # 1. Update Score
        if is_correct:
            st.session_state.score += 1
            
        # 2. Update Difficulty (Adaptive Logic)
        update_difficulty(is_correct)
        
        # 3. Store Data
        st.session_state.answers.append({
            "question": question_data['question'],
            "user_answer": user_choice,
            "correct": is_correct,
            "concept": question_data.get('concept'),
            "difficulty": st.session_state.current_difficulty
        })
        
        st.session_state.asked_concepts.append(question_data.get('concept', ''))
        st.session_state.show_answer = True
        st.session_state.force_rerun = True

    if st.session_state.force_rerun:
        st.session_state.force_rerun = False
        st.rerun()

def display_quiz():
    """Display the quiz interface"""
    
    # 1. Generate New Question if needed
    if st.session_state.current_question is None:
        with st.spinner(f"Generating {st.session_state.current_difficulty} question via LangChain..."):
            styles = st.session_state.user_inputs["Testing Structure"]["Question Styles"]
            st.session_state.current_question_style = random.choice(styles)
            
            st.session_state.current_question = get_question(
                st.session_state.user_inputs,
                st.session_state.asked_concepts,
                st.session_state.current_difficulty,
                st.session_state.current_question_style
            )
            st.session_state.answer_submitted = False
            st.session_state.timer_expired = False
            st.rerun()

    # 2. Show Question or Answer
    if not st.session_state.show_answer:
        display_question_with_timer()
    else:
        # Show Result
        q_data = st.session_state.current_question
        last_ans = st.session_state.answers[-1]
        
        st.subheader("Result")
        if last_ans['user_answer'] is None:
            st.warning("⏰ Time's up!")
        elif last_ans['correct']:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Incorrect. You chose: {last_ans['user_answer']}")
            
        st.info(f"**Correct Answer:** {q_data['correct_answer']}")
        st.markdown(f"**Explanation:** {q_data['explanation']}")
        
        # Next Button Logic
        total_q = st.session_state.user_inputs["Testing Structure"]["Question Count"]
        
        if st.session_state.question_count < total_q - 1:
            if st.button("Next Question ➡"):
                st.session_state.current_question = None
                st.session_state.show_answer = False
                st.session_state.question_count += 1
                st.rerun()
        else:
            st.balloons()
            st.success(f"🎉 Assessment Completed! Score: {st.session_state.score}/{total_q}")
            
            # Show Analytics
            st.subheader("Performance Summary")
            st.write("Concepts Covered:", ", ".join(set(st.session_state.asked_concepts)))
            
            if st.button("Start New Assessment"):
                initialize_session_state()
                st.rerun()

# --- 6. MAIN APP LOOP ---

def main():
    initialize_session_state()
    
    if not st.session_state.quiz_started:
        display_configuration_form()
    else:
        display_quiz()

if __name__ == "__main__":
    main()



# import streamlit as st 
# import os
# import json
# import time
# import random
# from datetime import datetime, timedelta
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate

# # Initialize LLM

# llm = ChatGroq(groq_api_key=st.secrets["groq"]["api_key"], model_name="groq/compound")
# # llm = ChatGroq(groq_api_key=st.secrets["groq"]["api_key"], model_name="Llama3-70b-8192")
# # groq/compound

# def initialize_session_state():
#     """Initialize or reset session state variables"""
#     if 'quiz_started' not in st.session_state:
#         st.session_state.quiz_started = False
#     if 'user_inputs' not in st.session_state:
#         st.session_state.user_inputs = None
#     if 'current_question' not in st.session_state:
#         st.session_state.current_question = None
#     if 'question_count' not in st.session_state:
#         st.session_state.question_count = 0
#     if 'score' not in st.session_state:
#         st.session_state.score = 0
#     if 'answers' not in st.session_state:
#         st.session_state.answers = []
#     if 'show_answer' not in st.session_state:
#         st.session_state.show_answer = False
#     if 'question_start_time' not in st.session_state:
#         st.session_state.question_start_time = None
#     if 'time_per_question' not in st.session_state:
#         st.session_state.time_per_question = 0
#     if 'asked_concepts' not in st.session_state:
#         st.session_state.asked_concepts = []
#     if 'current_difficulty' not in st.session_state:
#         st.session_state.current_difficulty = None
#     if 'current_question_style' not in st.session_state:
#         st.session_state.current_question_style = None
#     if 'timer_expired' not in st.session_state:
#         st.session_state.timer_expired = False
#     if 'answer_submitted' not in st.session_state:
#         st.session_state.answer_submitted = False
#     if 'force_rerun' not in st.session_state:
#         st.session_state.force_rerun = False

# def calculate_time_per_question(total_time, total_questions):
#     """Convert time string to seconds per question"""
#     if "hour" in total_time:
#         hours = int(total_time.split()[0])
#         total_seconds = hours * 3600
#     elif "mins" in total_time:
#         minutes = int(total_time.split()[0])
#         total_seconds = minutes * 60
#     else:  # 90 mins case
#         total_seconds = 90 * 60
    
#     return total_seconds / total_questions

# def get_question(user_input_dict, asked_concepts, current_difficulty, question_style):
#     """Generate a question based on user inputs"""
#     prompt_template = """
#     You are an expert question creator. Based on the following parameters:
#     - Testing Structure: {testing_structure}
#     - User Career Profile: {career_profile}
#     - Real World Alignment: {real_world_alignment}
#     - Already Asked Concepts: {asked_concepts}
#     - Current Difficulty: {current_difficulty}
#     - Question Style: {question_style}
    
#     Create a multiple-choice question (4 options, 1 correct answer) in JSON format with:
#     - question: The question text
#     - choices: List of 4 options
#     - correct_answer: The correct choice
#     - explanation: Brief explanation of the answer
#     - concept: The main concept being tested (1-2 words)
    
#     Rules:
#     1. The question should be unique and different from previous questions
#     2. Cover different concepts based on the already asked concepts list
#     3. Match the specified difficulty level
#     4. Use the specified question style
#     5. If difficulty is increasing, make subsequent questions more challenging
    
#     Return ONLY the JSON object, nothing else.
#     """
    
#     # Prepare the inputs for the prompt
#     testing_structure = "\n".join([f"{k}: {v}" for k, v in user_input_dict["Testing Structure"].items()])
#     career_profile = "\n".join([f"{k}: {v}" for k, v in user_input_dict["User Career Profile"].items()])
#     real_world_alignment = "\n".join([f"{k}: {v}" for k, v in user_input_dict["Real World Alignment"].items()])
    
#     # Format the prompt
#     prompt = prompt_template.format(
#         testing_structure=testing_structure,
#         career_profile=career_profile,
#         real_world_alignment=real_world_alignment,
#         asked_concepts=", ".join(asked_concepts[-5:]) if asked_concepts else "None",
#         current_difficulty=current_difficulty,
#         question_style=question_style
#     )
    
#     # Invoke LLM
#     response = llm.invoke(prompt)
    
#     try:
#         # Extract JSON from the response
#         start_idx = response.content.find('{')
#         end_idx = response.content.rfind('}') + 1
#         json_str = response.content[start_idx:end_idx]
#         data = json.loads(json_str)
        
#         # Validate the response has required fields
#         required_fields = ['question', 'choices', 'correct_answer']
#         for field in required_fields:
#             if field not in data:
#                 st.warning(f"Generated question missing required field: {field}")
#                 # Provide defaults for missing fields
#                 if field == 'question':
#                     data['question'] = "Sample question (generated due to missing content)"
#                 elif field == 'choices':
#                     data['choices'] = ["Option A", "Option B", "Option C", "Option D"]
#                 elif field == 'correct_answer':
#                     data['correct_answer'] = data['choices'][0] if data.get('choices') else "Option A"
        
#         # Ensure optional fields exist
#         if 'explanation' not in data:
#             data['explanation'] = "Explanation not provided"
#         if 'concept' not in data:
#             data['concept'] = "General"
            
#         return data
#     except (json.JSONDecodeError, ValueError) as e:
#         st.error(f"Failed to parse question: {e}")
#         # Return a default question if parsing fails
#         return {
#             'question': "Sample question (generated due to error)",
#             'choices': ["Option A", "Option B", "Option C", "Option D"],
#             'correct_answer': "Option A",
#             'explanation': "This question was generated because the system encountered an error",
#             'concept': "Error Handling"
#         }

# def display_configuration_form():
#     """Display the initial configuration form"""
#     st.title("🧪 Testing & Career Skill Assessment Configuration")

#     with st.form("config_form"):
#         st.header("1. Testing Structure")
#         test_name = st.text_input("Test Name", placeholder="Enter test name")
#         question_count = st.slider("Question Count", min_value=1, max_value=50, value=10, step=1)
#         time_limit = st.selectbox("Time Limit", ["30 mins", "1 hour", "90 mins"])
#         question_format = st.multiselect("Question Format", ["MCQ", "Code Output", "Coding Snippet", "Diagram", "Case Study"], default=["MCQ"])
#         question_style = st.multiselect("Question Style", ["Descriptive and explained", "Briefly explained", "Innovative Scenario based", "Debugging based"], default=["Descriptive and explained", "Innovative Scenario based"])
#         difficulty = st.selectbox("Initial Difficulty Level", ["Beginner", "Intermediate", "Senior Level", "Hiring Challenge"])
#         negative_marking = st.checkbox("Enable Negative Marking")
#         neg_mark_value = st.number_input("Negative Marking Value (e.g., -0.25)", value=-0.25) if negative_marking else 0
#         adaptive_progression = st.toggle("Adaptive Progression (ON = Difficulty increases)", value=True)

#         st.divider()

#         st.header("2. User Career Profile")
#         target_role = st.multiselect("Target Role", ["Full Stack Developer", "Data Scientist", "ML Engineer", "SDE II"])
#         experience = st.slider("Years of Experience", 0, 10, 1)
#         current_industry = st.selectbox("Current Industry", ["FinTech", "EdTech", "SaaS", "E-commerce", "Healthcare"])
#         target_industry = st.selectbox("Target Industry", ["AI Research", "Product Dev", "Finance", "GovTech"])
#         target_companies = st.multiselect("Target Companies", ["Google", "Meta", "Flipkart", "Infosys", "Razorpay", "Startups"])
#         tech_stack = st.text_input("Tech Stack Worked On (comma-separated)", placeholder="e.g., React, Django, SQL, AWS, PyTorch")
#         interested_tech = st.text_input("Interested Technologies (comma-separated)", placeholder="e.g., Web3, LLMs, Microservices, Big Data")
#         learning_goals = st.multiselect("Learning Goals", ["Job Prep", "Skill Upskilling", "Certification", "Mock Interviews"])

#         st.divider()

#         st.header("3. Real World Alignment")
#         sim_round_type = st.selectbox("Simulated Round Type", ["Online Test", "System Design", "Code Pairing", "Take-Home"])
#         interview_style = st.radio("Interview Style", ["Simulated", "Educational", "Speed Test"], index=0)
#         company_sim = st.selectbox("Company Simulation", ["Meta", "TCS", "Ola", "etc."])
#         role_level = st.selectbox("Role Level", ["Intern", "Entry", "Mid-Level", "Senior", "Lead"])

#         submitted = st.form_submit_button("Start Quiz")

#         if submitted:
#             user_inputs = {
#                 "Testing Structure": {
#                     "Test Name": test_name,
#                     "Question Count": question_count,
#                     "Time Limit": time_limit,
#                     "Question Format": question_format,
#                     "Question Styles": question_style,
#                     "Initial Difficulty": difficulty,
#                     "Negative Marking": negative_marking,
#                     "Negative Mark Value": neg_mark_value,
#                     "Adaptive Progression": adaptive_progression
#                 },
#                 "User Career Profile": {
#                     "Target Role": target_role,
#                     "Years of Experience": experience,
#                     "Current Industry": current_industry,
#                     "Target Industry": target_industry,
#                     "Target Companies": target_companies,
#                     "Tech Stack": [tech.strip() for tech in tech_stack.split(",") if tech.strip()],
#                     "Interested Technologies": [tech.strip() for tech in interested_tech.split(",") if tech.strip()],
#                     "Learning Goals": learning_goals
#                 },
#                 "Real World Alignment": {
#                     "Simulated Round Type": sim_round_type,
#                     "Interview Style": interview_style,
#                     "Company Simulation": company_sim,
#                     "Role Level": role_level
#                 }
#             }
            
#             st.session_state.user_inputs = user_inputs
#             st.session_state.quiz_started = True
#             st.session_state.time_per_question = calculate_time_per_question(
#                 user_inputs["Testing Structure"]["Time Limit"],
#                 user_inputs["Testing Structure"]["Question Count"]
#             )
#             st.session_state.current_difficulty = user_inputs["Testing Structure"]["Initial Difficulty"]
#             st.rerun()

# def format_time(seconds):
#     """Format seconds into MM:SS format"""
#     return str(timedelta(seconds=int(seconds)))[2:]

# def update_difficulty():
#     """Increase difficulty if adaptive progression is on"""
#     if st.session_state.user_inputs["Testing Structure"]["Adaptive Progression"]:
#         difficulty_levels = ["Beginner", "Intermediate", "Senior Level", "Hiring Challenge"]
#         current_index = difficulty_levels.index(st.session_state.current_difficulty)
#         if current_index < len(difficulty_levels) - 1:
#             st.session_state.current_difficulty = difficulty_levels[current_index + 1]

# def display_question_with_timer():
#     """Display the question with a real-time countdown timer"""
#     question_data = st.session_state.current_question
#     total_questions = st.session_state.user_inputs["Testing Structure"]["Question Count"]
#     time_per_question = st.session_state.time_per_question
    
#     # Create layout columns
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Progress", f"{st.session_state.question_count + 1}/{total_questions}")
#     with col2:
#         st.metric("Score", st.session_state.score)
#     with col3:
#         timer_placeholder = st.empty()
#     with col4:
#         if st.button("End Test", type="primary"):
#             st.session_state.question_count = total_questions - 1
#             st.session_state.show_answer = True
#             st.session_state.timer_expired = True
#             st.rerun()
    
#     # Display the question
#     st.subheader("Question:")
#     st.markdown(f"**{question_data.get('question', 'Question text not available')}**")
#     st.caption(f"Difficulty: {st.session_state.current_difficulty} | Style: {st.session_state.current_question_style}")
    
#     # Start the timer
#     start_time = time.time()
#     end_time = start_time + time_per_question
    
#     # Create form for answer submission
#     answer_form = st.form(key="answer_form")
    
#     # Ensure choices exist and are valid
#     choices = question_data.get('choices', ["Option A", "Option B", "Option C", "Option D"])
#     if not isinstance(choices, list) or len(choices) < 1:
#         choices = ["Option A", "Option B", "Option C", "Option D"]
    
#     user_choice = answer_form.radio(
#         "Select your answer:",
#         choices,
#         index=None,
#         key=f"question_{st.session_state.question_count}"
#     )
#     submitted = answer_form.form_submit_button("Submit Answer")
    
#     # Run the timer
#     while True:
#         current_time = time.time()
#         remaining_time = max(0, end_time - current_time)
#         timer_placeholder.metric("Time Remaining", format_time(remaining_time))
        
#         # Check if time is up or answer submitted
#         if current_time >= end_time:
#             st.session_state.timer_expired = True
#             st.session_state.show_answer = True
#             st.session_state.answers.append({
#                 "question": question_data.get('question', 'Unknown question'),
#                 "user_answer": None,
#                 "correct": False,
#                 "time_taken": time_per_question,
#                 "concept": question_data.get('concept', 'Unknown')
#             })
#             st.session_state.force_rerun = True
#             break
            
#         if submitted or st.session_state.force_rerun:
#             break
            
#         # Small delay to prevent excessive rerenders
#         time.sleep(0.1)
    
#     if submitted and user_choice is not None:
#         st.session_state.answer_submitted = True
#         time_taken = time.time() - start_time
        
#         # Handle case where correct_answer might be missing
#         correct_answer = question_data.get('correct_answer')
#         if correct_answer is None or correct_answer not in choices:
#             st.warning("Correct answer may not exist in options. This will be marked as correct.")
#             is_correct = True
#             correct_answer = user_choice
#         else:
#             is_correct = user_choice == correct_answer
        
#         st.session_state.answers.append({
#             "question": question_data.get('question', 'Unknown question'),
#             "user_answer": user_choice,
#             "correct": is_correct,
#             "time_taken": time_taken,
#             "concept": question_data.get('concept', 'Unknown')
#         })
        
#         if is_correct:
#             st.session_state.score += 1
#             if st.session_state.user_inputs["Testing Structure"]["Adaptive Progression"]:
#                 update_difficulty()
        
#         if 'concept' in question_data:
#             st.session_state.asked_concepts.append(question_data['concept'])
        
#         st.session_state.show_answer = True
#         st.session_state.force_rerun = True
    
#     if st.session_state.force_rerun:
#         st.session_state.force_rerun = False
#         st.rerun()

# def display_quiz():
#     """Display the quiz interface"""
#     st.title("📝 Quiz Time!")
    
#     # Generate new question if needed
#     if st.session_state.current_question is None:
#         with st.spinner("Generating your question..."):
#             selected_style = random.choice(st.session_state.user_inputs["Testing Structure"]["Question Styles"])
#             st.session_state.current_question_style = selected_style
            
#             st.session_state.current_question = get_question(
#                 st.session_state.user_inputs,
#                 st.session_state.asked_concepts,
#                 st.session_state.current_difficulty,
#                 st.session_state.current_question_style
#             )
#             st.session_state.answer_submitted = False
#             st.session_state.timer_expired = False
#             st.rerun()
    
#     if not st.session_state.show_answer:
#         display_question_with_timer()
#     else:
#         # Show answer and explanation
#         question_data = st.session_state.current_question
#         last_answer = st.session_state.answers[-1]
        
#         if last_answer['user_answer'] is not None:
#             if last_answer['correct']:
#                 st.success(f"✅ Correct! ")
#             else:
#                 st.error(f"❌ Incorrect! ")
#                 correct_answer = question_data.get('correct_answer', 'Correct answer not specified')
#                 st.error(f"The correct answer is: {correct_answer}")
#         else:
#             st.warning("⏰ Time's up! Question not attempted.")
#             correct_answer = question_data.get('correct_answer', 'Correct answer not specified')
#             st.info(f"The correct answer is: {correct_answer}")
        
#         # Display explanation if it exists
#         explanation = question_data.get('explanation')
#         if explanation:
#             st.markdown(f"**Explanation:** {explanation}")
        
#         # Display concept if it exists
#         concept = question_data.get('concept')
#         if concept:
#             st.markdown(f"**Concept tested:** {concept}")
        
#         total_questions = st.session_state.user_inputs["Testing Structure"]["Question Count"]
#         if st.session_state.question_count < total_questions - 1:
#             if st.button("Next Question", key="next_question"):
#                 with st.spinner("Generating next question..."):
#                     st.session_state.current_question = None
#                     st.session_state.show_answer = False
#                     st.session_state.question_count += 1
#                     st.rerun()
#         else:
#             st.balloons()
#             st.success(f"🎉 Quiz completed! Your final score: {st.session_state.score}/{total_questions}")
            
#             st.subheader("Concepts Covered:")
#             concepts = [ans.get('concept', 'Unknown') for ans in st.session_state.answers]
#             st.write(", ".join(set(concepts)))
            
#             if st.button("Start New Quiz"):
#                 initialize_session_state()
#                 st.rerun()

# def main():
#     """Main app function"""
#     initialize_session_state()
    
#     if not st.session_state.quiz_started:
#         display_configuration_form()
#     else:
#         display_quiz()

# if __name__ == "__main__":
#     main()
