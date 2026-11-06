
import json
from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# Task 1: Setup and helper function
# --------------------------------------------------

# Load the API key from the .env file
load_dotenv()

# Create the OpenAI client
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    """Send messages to the model and return its response."""

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )

    return response.choices[0].message.content


# System prompt for the job application helper
SYSTEM_PROMPT = """
You are a helpful job application coach who supports people
changing careers and applying for new jobs.

Help users improve resume bullet points, prepare cover letter
openings, and answer questions related to job applications.

Stay focused on job application materials and career transitions.

Do not invent work experience, qualifications, skills, or achievements
that the user has not provided.

Always remind users to review and edit AI-generated content before
submitting it to an employer.

You may not know the specific expectations or industry norms
for every job or employer. Encourage users to use their own
judgment and check relevant industry expectations.
"""

# I included instructions not to invent experience or qualifications
# because job application materials must accurately represent the user.


# --------------------------------------------------
# Task 2: Resume bullet point rewriter
# --------------------------------------------------

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    """Rewrite resume bullet points and return structured results."""

    # Put the user's bullet points into a clearly marked section
    bullet_text = "\n".join(f"- {bullet}" for bullet in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.

    Rewrite each resume bullet point to make it clearer and more
    professional. Use strong action verbs where appropriate.

    IMPORTANT:
    - Keep each improved bullet faithful to the original.
    - Do not invent numbers, results, tools, achievements, or impact.
    - Do not add information that the user did not provide.
    - If details are missing, improve the wording without inventing them.

    Return ONLY valid JSON as a list.
    Do not include Markdown fences or extra text.

    Each item must have two keys:
    "original" and "improved".

    Original bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = get_completion(messages)

    # Remove Markdown code fences if the model includes them
    cleaned_response = response.strip()

    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response.split("\n", 1)[1]
        cleaned_response = cleaned_response.rsplit("```", 1)[0].strip()

    print("\nCleaned response:")
    print(cleaned_response)

    # Convert the JSON response into a Python list
    results = json.loads(cleaned_response)

    # Check that the response is a list
    if not isinstance(results, list):
        raise ValueError("Expected a JSON list from the model.")

    # Check that every item contains original and improved keys
    for item in results:
        if not isinstance(item, dict):
            raise ValueError(f"Expected a dictionary, but received: {item}")

        if "original" not in item or "improved" not in item:
            raise ValueError(f"Missing required keys: {item}")

        print("\nOriginal:", item["original"])
        print("\nImproved:", item["improved"])
        print("-" * 50)

    return results


# --------------------------------------------------
# Task 2: Test the bullet point rewriter
# --------------------------------------------------

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)


# Observation:
# The original bullet points were weak because they were too general.
# The model suggested clearer wording and stronger action verbs.
# The improved bullets sound more professional while keeping the
# original meaning and avoiding unsupported achievements.


# --------------------------------------------------
# Task 3: Cover Letter Generator
# --------------------------------------------------

def generate_cover_letter(job_title: str, background: str) -> str:
    """Generate a cover letter opening using few-shot prompting."""

    messages = [
        {
            "role": "system",
            "content": """
            You are a professional cover letter writing assistant.

            Write a strong and professional cover letter opening
            for someone changing careers.

            Do not invent experience, qualifications, or skills.
            Keep the writing confident, specific, and natural.
            Avoid generic phrases and clichés.
            """
        },
        {
            "role": "user",
            "content": """
            Write a cover letter opening for a Junior Data Analyst.

            Background:
            Three years of experience as a teacher.
            Experience organizing student performance data
            and using spreadsheets to track progress.
            """
        },
        {
            "role": "assistant",
            "content": """
            My experience as a teacher has strengthened my ability
            to organize information, recognize patterns, and communicate
            insights clearly. I am interested in bringing these skills
            to a Junior Data Analyst role, where I can contribute to
            data-driven problem solving while continuing to develop
            my technical abilities.
            """
        },
        {
            "role": "user",
            "content": """
            Write a cover letter opening for an IT Support Specialist.

            Background:
            Four years of experience in customer service.
            Regularly helped customers resolve questions and problems.
            """
        },
        {
            "role": "assistant",
            "content": """
            My background in customer service has developed my ability
            to listen carefully, understand problems, and communicate
            solutions clearly. I am excited to apply these strengths
            in an IT Support Specialist role and continue building
            my technical support skills.
            """
        },
        {
            "role": "user",
            "content": f"""
            Write a cover letter opening for this position:

            Job title: {job_title}

            Background:
            {background}

            Requirements:
            - Write 3 to 5 sentences.
            - Be confident, specific, and professional.
            - Avoid clichés and generic introductions.
            - Do not invent experience or qualifications.
            - Use only the background information provided.
            """
        }
    ]

    response = get_completion(messages)

    print("\nGenerated Cover Letter Opening:")
    print(response)

    return response


# --------------------------------------------------
# Task 3: Test the cover letter generator
# --------------------------------------------------

job_title = "Junior Data Engineer"

background = """
Five years of experience as a middle school math teacher;
recently completed a Python course and built data pipelines
using Pandas.
"""

generate_cover_letter(job_title, background)


# Few-shot prompting observation:
# I provided two examples of cover letter openings for different career transitions. These examples show the model the desired tone, structure, and level of detail. This helps guide the model when generating an opening for a new job title and background.


# --------------------------------------------------
# Task 4: Content Moderation
# --------------------------------------------------

def is_safe(text: str) -> bool:
    """Check whether text is flagged by the moderation API."""

    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = response.results[0].flagged

    if flagged:
        print(
            "This request cannot be processed. "
            "Please keep the conversation safe and respectful."
        )
        return False

    return True


# --------------------------------------------------
# Task 4: Test Content Moderation
# --------------------------------------------------

safe_text = "I am interested in applying for a Junior Data Engineer job."

flagged_text = "I want to hurt someone."

print("\nTask 4: Content Moderation")

print("Safe input result:", is_safe(safe_text))
print("Flagged input result:", is_safe(flagged_text))


# --------------------------------------------------
# Task 5: Interactive Job Application Chatbot
# --------------------------------------------------

def run_chatbot():
    """Run an interactive job application chatbot."""

    # Start conversation history with the system prompt
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    print("\nWelcome to the Job Application Helper!")
    print("You can ask questions about resumes and job applications.")
    print("Type 'quit' or 'exit' to end the conversation.")

    while True:

        user_input = input("\nYou: ").strip()

        # Exit the chatbot
        if user_input.lower() in ["quit", "exit"]:
            print("Thank you for using the Job Application Helper!")
            break

        # Skip empty messages
        if not user_input:
            continue

        # Check moderation before processing the request
        if not is_safe(user_input):
            continue

        lower_input = user_input.lower()

        # ------------------------------------------
        # Resume bullet point request
        # ------------------------------------------

        if "bullet" in lower_input or "resume" in lower_input:

            print("\nEnter your resume bullet points.")
            print("Enter one bullet point per line.")
            print("Type DONE when finished.")

            bullets = []

            while True:
                bullet = input("> ").strip()

                if bullet.upper() == "DONE":
                    break

                if bullet:
                    bullets.append(bullet)

            if not bullets:
                print("No bullet points were entered.")
                continue

            # Moderate the collected bullet points too
            if not is_safe("\n".join(bullets)):
                continue

            results = rewrite_bullets(bullets)

            # Save the request and result in conversation history
            messages.append({
                "role": "user",
                "content": user_input + "\n" + "\n".join(bullets)
            })

            messages.append({
                "role": "assistant",
                "content": json.dumps(results)
            })

        # ------------------------------------------
        # Cover letter request
        # ------------------------------------------

        elif "cover letter" in lower_input:

            print("\nLet's prepare your cover letter opening.")

            job_title = input("Enter the job title: ").strip()

            background = input(
                "Enter your experience and background: "
            ).strip()

            if not job_title or not background:
                print("Please provide both the job title and background.")
                continue

            # Moderate the collected information
            collected_text = job_title + "\n" + background

            if not is_safe(collected_text):
                continue

            result = generate_cover_letter(
                job_title,
                background
            )

            # Save the request and result in conversation history
            messages.append({
                "role": "user",
                "content": (
                    user_input
                    + "\nJob title: " + job_title
                    + "\nBackground: " + background
                )
            })

            messages.append({
                "role": "assistant",
                "content": result
            })

        # ------------------------------------------
        # Regular conversation
        # ------------------------------------------

        else:

            messages.append({
                "role": "user",
                "content": user_input
            })

            response = get_completion(messages)

            print("\nAssistant:", response)

            # Save assistant response in conversation history
            messages.append({
                "role": "assistant",
                "content": response
            })


# --------------------------------------------------
# Task 5: Run the chatbot
# --------------------------------------------------

run_chatbot()



# --------------------------------------------------
# Task 6: Reflection on AI Ethics
# --------------------------------------------------

# AI tools can have bias because their training data may favor
# certain communication styles, industries, or cultures.
#
# Users should not submit AI-generated resumes or cover letters
# without reviewing them because the model may add inaccurate
# information or make claims that do not reflect their experience.
#
# One professional guardrail is to verify every skill, qualification,
# and achievement before including AI-generated content in a job
# application. Users should also make sure the final writing
# represents their own experience and voice.
