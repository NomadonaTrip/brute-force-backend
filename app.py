from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from groq import Groq


app = Flask(__name__)
CORS(app)




# Initialize Groq client
def get_groq_client():
    return Groq(api_key=os.environ.get('GROQ_API_KEY'))

@app.route('/api/generate-response', methods=['POST'])
def generate_response():
    """Generate AI prospect response based on user's sales pitch"""
    try:
        data = request.json
        
        round_num = data.get('round', 1)
        user_transcript = data.get('userTranscript', '')
        prospect_name = data.get('prospectName', '')
        prospect_type = data.get('prospectType', '')
        prospect_context = data.get('prospectContext', '')
        prep_answers = data.get('prepAnswers', {})
        
        # Build the prompt for Groq
        system_prompt = f"""You are roleplaying as {prospect_name}, a {prospect_type}. 

Context: {prospect_context}

You are in a sales conversation with a web designer. This is round {round_num} of the conversation.

Your behavior should be realistic and challenging:
- Round 1: You're skeptical. The salesperson hasn't earned your trust yet. Give vague, guarded responses.
- Round 2: If they ask good discovery questions, open up a bit. Mention your challenges but don't volunteer solutions yet.
- Round 3+: If they've uncovered your real pain and shown they understand your business, you become interested and ask them what they would suggest.

Key traits:
- You're busy and won't tolerate generic pitches
- You want to understand WHY you should care, not just WHAT they're selling
- You only "pull" (ask for their solution) when they've proven they understand your business problem
- You're realistic - you mention trade shows, referrals, and current challenges naturally

IMPORTANT: 
- If the salesperson asks good discovery questions about your buyers, competitors, or pain points, respond positively
- If they jump to solutions without understanding your problem, give them a brush-off
- Only ask "What would you suggest?" or "How would you help with that?" if they've truly uncovered urgent pain
- Keep responses to 2-3 sentences maximum

Respond ONLY as the prospect. Do not break character.
"""

        user_message = f"Salesperson says: {user_transcript}"
        
        # Call Groq API
        chat_completion = get_groq_client().chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-70b-versatile",
            temperature=0.7,
            max_tokens=200
        )
        
        prospect_response = chat_completion.choices[0].message.content
        
        # Analyze the conversation to determine evaluation
        evaluation = evaluate_conversation(
            round_num, 
            user_transcript, 
            prospect_response,
            prospect_type
        )
        
        return jsonify({
            'response': prospect_response,
            'evaluation': evaluation['evaluation'],
            'gotPull': evaluation['gotPull'],
            'warmth': evaluation['warmth']
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


def evaluate_conversation(round_num, user_transcript, prospect_response, prospect_type):
    """Evaluate the quality of the sales conversation"""
    
    eval_prompt = f"""You are a sales trainer evaluating a sales conversation.

Prospect type: {prospect_type}
Round: {round_num}

Salesperson said: "{user_transcript}"
Prospect responded: "{prospect_response}"

Evaluate the salesperson's performance on these criteria:

1. Did they uncover REAL pain (not just surface-level problems)?
2. Did they create URGENCY (help prospect see cost of inaction)?
3. Did they establish CREDIBILITY (show they understand the business)?
4. Did the prospect "PULL" (ask for a solution/proposal)?

Provide evaluation in this EXACT JSON format:
{{
  "uncoveredPain": true/false,
  "createdUrgency": true/false,
  "establishedCredibility": true/false,
  "gotPull": true/false,
  "warmth": "HOT/WARM/COOL/COLD",
  "feedback": [
    "Specific feedback point 1",
    "Specific feedback point 2",
    "Specific feedback point 3"
  ]
}}

Warmth guide:
- HOT: Prospect is asking for solution, ready to move forward
- WARM: Prospect acknowledges pain and is engaged, but not ready yet
- COOL: Prospect is polite but not seeing urgent need
- COLD: Prospect is dismissive or giving brush-offs

Return ONLY the JSON, no other text."""

    try:
        eval_completion = get_groq_client().chat.completions.create(
            messages=[
                {"role": "user", "content": eval_prompt}
            ],
            model="llama-3.1-70b-versatile",
            temperature=0.3,
            max_tokens=500
        )
        
        eval_text = eval_completion.choices[0].message.content
        
        # Parse JSON response
        import json
        eval_text = eval_text.strip()
        if eval_text.startswith('```'):
            eval_text = eval_text.split('```')[1]
            if eval_text.startswith('json'):
                eval_text = eval_text[4:]
        
        eval_data = json.loads(eval_text.strip())
        
        return {
            'evaluation': {
                'uncoveredPain': eval_data.get('uncoveredPain', False),
                'createdUrgency': eval_data.get('createdUrgency', False),
                'establishedCredibility': eval_data.get('establishedCredibility', False),
                'feedback': eval_data.get('feedback', [])
            },
            'gotPull': eval_data.get('gotPull', False),
            'warmth': eval_data.get('warmth', 'COLD')
        }
        
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return {
            'evaluation': {
                'uncoveredPain': False,
                'createdUrgency': False,
                'establishedCredibility': False,
                'feedback': [
                    f"Error evaluating conversation: {str(e)}",
                    "Try again with more specific discovery questions"
                ]
            },
            'gotPull': False,
            'warmth': 'COLD'
        }

@app.route('/')
def home():
    return "Backend is running! Use /health endpoint to check status."

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)