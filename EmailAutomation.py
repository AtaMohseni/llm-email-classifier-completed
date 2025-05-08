# Configuration and imports
import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Sample email dataset
sample_emails = [
    {
        "id": "001",
        "from": "angry.customer@example.com",
        "subject": "Broken product received",
        "body": "I received my order #12345 yesterday but it arrived completely damaged. This is unacceptable and I demand a refund immediately. This is the worst customer service I've experienced.",
        "timestamp": "2024-03-15T10:30:00Z"
    },
    {
        "id": "002",
        "from": "curious.shopper@example.com",
        "subject": "Question about product specifications",
        "body": "Hi, I'm interested in buying your premium package but I couldn't find information about whether it's compatible with Mac OS. Could you please clarify this? Thanks!",
        "timestamp": "2024-03-15T11:45:00Z"
    },
    {
        "id": "003",
        "from": "happy.user@example.com",
        "subject": "Amazing customer support",
        "body": "I just wanted to say thank you for the excellent support I received from Sarah on your team. She went above and beyond to help resolve my issue. Keep up the great work!",
        "timestamp": "2024-03-15T13:15:00Z"
    },
    {
        "id": "004",
        "from": "tech.user@example.com",
        "subject": "Need help with installation",
        "body": "I've been trying to install the software for the past hour but keep getting error code 5123. I've already tried restarting my computer and clearing the cache. Please help!",
        "timestamp": "2024-03-15T14:20:00Z"
    },
    {
        "id": "005",
        "from": "business.client@example.com",
        "subject": "Partnership opportunity",
        "body": "Our company is interested in exploring potential partnership opportunities with your organization. Would it be possible to schedule a call next week to discuss this further?",
        "timestamp": "2024-03-15T15:00:00Z"
    }
]

class EmailProcessor:
    def __init__(self):
        """Initialize the email processor with OpenAI API key."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Define valid categories
        self.valid_categories = {
            "complaint", "inquiry", "feedback",
            "support_request", "other"
        }

    def classify_email(self, email: Dict) -> Optional[str]:
        """
        Classify an email using LLM.
        Returns the classification category or None if classification fails.
        
        TODO: 
        1. Design and implement the classification prompt
        2. Make the API call with appropriate error handling
        3. Validate and return the classification
        """
        # define the classification task as a function format to be included in prompt. 
        # make it required to extract sentiment variable (i.e. not an optional variable)
        functions = [
            {
                "name": "classification",
                "description": "classify the email sentiment with given categories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sentiment": {
                            "type": "string",
                            "description": f"""sentiment of the email, should be one of {', '.join(['"'+d+'"' for d in self.valid_categories])}""",
                            "enum": [i for i in self.valid_categories],
                        },
                        
                    },"required": ["sentiment"]
                    
                },
            }
        ]
        
        # define OpenAI message format
        messages = [
            {
                "role": "user",
                "content": email["body"]
            }
        ]
        
        
        try:
            # LLM call, passing message and function info. Also make it required to use function calling
            response = self.client.chat.completions.create(model="gpt-3.5-turbo",messages=messages,functions=functions, function_call={"name": "classification"})
            # check if model refuse to response to API call
            if response.choices[0].message.refusal == None:
                # check if the output is a valid json format
                try:
                    classification_result = json.loads(response.choices[0].message.function_call.arguments)['sentiment']
                    # check if the retruned category is one of specified category
                    if classification_result in self.valid_categories:
                        return classification_result
                    else:
                        logger.info("Invalid email Category")
                        return None
                except:
                    logger.info("Not valid JSON format")
                    return None
            else:
                logger.info("refused response by model")
                return None
        except Exception as e:
            logging.exception(f"Error On LLM API Call: {e}")
            return None
        

    def generate_response(self, email: Dict, classification: str) -> Optional[str]:
        """
        Generate an automated response based on email classification.
        
        TODO:
        1. Design the response generation prompt
        2. Implement appropriate response templates
        3. Add error handling
        """

        templates = {}
        # define a prompt template for each category
        complaint_template = f"""You are a great customer service agent \
        who can reply to a complaint emails promptly. Based on the \
        following recieved email, write an appropriate email response \
        to the customer, and let the customer know that we recieved the complain. \
        Let the customer know that a member of team will reach to him soon to address the issue. \
        Make sure to use the polite and understanding tone. \
        Do not use placeholder in email, use "customer" for the recipient name. \
        Only Sign the response email with "Customer Service Team".  
        
        Here is the recieved email information:
        customer email body: {email['body']}
        """
        templates["complaint"]  = complaint_template
        
        inquiry_template = f"""You are a great customer service agent \
        who can reply to inquery emails. Based on the following recieved \
        email, write an appropriate email to customer confirming that we \
        recieved the inquery, and it rent to a right team to address the \
        inquery. They will reply to the inquery with detailed information \
        as soon as possible. Make sure to use the polite and understanding tone.\
        Do not use placeholder in email, use "customer" for the recipient name. \
        Only Sign the response email with "Customer Service Team". 
        
        Here is the recieved email information:
        customer email body: {email['body']}
        """
        templates["inquiry"] = inquiry_template
        
        feedback_template = f"""You are a great customer service agent \
        who can reply to a feedback email. based on the following recieved \
        email, write an appropriate email to customer confirming that we \
        recieved the feedback, and appreciate them for the feedback, \
        also let them know that their feedback is valuable to us. \
        Make sure to use the polite and understanding tone.\
        Do not use placeholder in email, use "customer" for the recipient name. \
        Only Sign the response email with "Customer Service Team". 
        
        here is the recieved email information:
        customer email body: {email['body']} 
        """
        templates["feedback"] = feedback_template
        
        support_request_template = f"""You are a great customer service agent \
        who can reply to a support request email. Based on the following recieved \
        email, write an appropriate response email to customer confirming that we \
        recieved the support request, it sent to a specialized team to address \
        the request, and they will contact the customer as soon as possible with \
        detailed and accurate information. \
        Make sure to use the polite and understanding tone.\
        Do not use placeholder in email, use "customer" for the recipient name. \
        Only Sign the response email with "Customer Service Team". 
        
        Here is the recieved email information:
        customer email body: {email['body']} 
        """
        templates["support_request"] = support_request_template
        
        other_template = f"""You are a great customer service agent \
        who can reply to recieving emails. For the following recieved email we couldnt assign a category to it. \
        The recieved email is NOT {', or '.join(['"'+d+'"' for d in self.valid_categories.difference({"other"}) ])}. \
        Based on the following recieved email, write an appropriate response email to customer confirming that we \
        recieved the email, and thanks for contacting us and we will get back to customer if it is needed. \
        Do Not include any other information.\
        Make sure to use the polite and understanding tone.\
        Do not use placeholder in email, use "customer" for the recipient name. \
        Only Sign the response email with "Customer Service Team". 
        
        Here is the recieved email information:
        customer email body: {email['body']} 
        """
        templates["other"] = other_template
        
        # define OpenAI message
        messages = [
            {
                "role": "user",
                "content": templates[classification]
            }
        ]        
        
        def check_for_placeholder(generated_email):
            """ function to check if generated email has a placeholder"""
            if ("[" in generated_email and "]" in generated_email) or ("{{" in generated_email and "}}" in generated_email):
                logger.info("generated response email has a place holder")
                return False
            else:
                return True

        try:
            # LLM API call
            response = self.client.chat.completions.create(model="gpt-3.5-turbo",messages=messages)
            #check if model refuse to response to API call
            if response.choices[0].message.refusal == None:
                email_content = response.choices[0].message.content
                # check if response content is empty string
                if email_content.strip():
                    # check if the genereated response has place holder (i.e. [] or {{}})
                    if check_for_placeholder(email_content):
                        response_str = response.choices[0].message.content
                        return response_str
                    else:
                        return None
                else:
                    logger.info("The generated email is empty or contains only whitespace")
                    return None
            else:
                logger.info("refused response by model")
                return None
                
        except Exception as e:
            logging.exception(f"Error On LLM API Call: {e}")
            return None

class EmailAutomationSystem:
    def __init__(self, processor: EmailProcessor):
        """Initialize the automation system with an EmailProcessor."""
        self.processor = processor
        self.response_handlers = {
            "complaint": self._handle_complaint,
            "inquiry": self._handle_inquiry,
            "feedback": self._handle_feedback,
            "support_request": self._handle_support_request,
            "other": self._handle_other
        }

    def process_email(self, email: Dict) -> Dict:
        """
        Process a single email through the complete pipeline.
        Returns a dictionary with the processing results.
        
        TODO:
        1. Implement the complete processing pipeline
        2. Add appropriate error handling
        3. Return processing results
        
        """
        processing_result = {}
        processing_result["email_id"] = email["id"]
        # classify the email
        classification_str = self.processor.classify_email(email)
        
        # check if the classification is a valid classification category
        if classification_str is not None and classification_str in self.response_handlers.keys():
            
            processing_result["classification"]=classification_str
            self.response_handlers[classification_str](email)
            
            # generate response
            response = self.processor.generate_response(email, classification_str)
            
            # check the response
            if response is not None:
                processing_result['response_sent'] = response
                processing_result['success'] = "YES"
            else:
                logger.info("Invalid email response message")
                processing_result['response_sent'] = 'None'
                processing_result['success'] = 'None'
        else: 
            logger.info("invalid email category")
            processing_result["classification"]= 'None'
            processing_result['success'] = 'None'
            processing_result['response_sent'] = 'None'
        
        return processing_result

    def _handle_complaint(self, email: Dict):
        """
        Handle complaint emails.
        TODO: Implement complaint handling logic
        """
        email_id = email["id"]                         
        logger.info(f"handle complaint for {email_id}")                         
        pass
        

    def _handle_inquiry(self, email: Dict):
        """
        Handle inquiry emails.
        TODO: Implement inquiry handling logic
        """
        email_id = email["id"] 
        logger.info(f"handle inquiry for {email_id}")                          
        pass

    def _handle_feedback(self, email: Dict):
        """
        Handle feedback emails.
        TODO: Implement feedback handling logic
        """
        email_id = email["id"]
        logger.info(f"handle feedback for {email_id}") 
        pass
        

    def _handle_support_request(self, email: Dict):
        """
        Handle support request emails.
        TODO: Implement support request handling logic
        """
        email_id = email["id"]
        logger.info(f"handle support request for {email_id}")                          
        pass
        

    def _handle_other(self, email: Dict):
        """
        Handle other category emails.
        TODO: Implement handling logic for other categories
        """
        email_id = email["id"]                         
        logger.info(f"handle other request for {email_id}")
        pass
        

# Mock service functions
def send_complaint_response(email_id: str, response: str):
    """Mock function to simulate sending a response to a complaint"""
    logger.info(f"Sending complaint response for email {email_id}")
    # In real implementation: integrate with email service


def send_standard_response(email_id: str, response: str):
    """Mock function to simulate sending a standard response"""
    logger.info(f"Sending standard response for email {email_id}")
    # In real implementation: integrate with email service


def create_urgent_ticket(email_id: str, category: str, context: str):
    """Mock function to simulate creating an urgent ticket"""
    logger.info(f"Creating urgent ticket for email {email_id}")
    # In real implementation: integrate with ticket system


def create_support_ticket(email_id: str, context: str):
    """Mock function to simulate creating a support ticket"""
    logger.info(f"Creating support ticket for email {email_id}")
    # In real implementation: integrate with ticket system


def log_customer_feedback(email_id: str, feedback: str):
    """Mock function to simulate logging customer feedback"""
    logger.info(f"Logging feedback for email {email_id}")
    # In real implementation: integrate with feedback system


def run_demonstration():
    """Run a demonstration of the complete system."""
    # Initialize the system
    processor = EmailProcessor()
    automation_system = EmailAutomationSystem(processor)

    # Process all sample emails
    results = []
    for email in sample_emails:
        # verify the email format first
        verified_email = False
        #check if email is in Dict format
        if isinstance(email, dict):
            # check if email has correct key(s)
            if "id" in email.keys() and "body" in email.keys():
                #check if email body is non-empty
                if email["body"].strip():
                    verified_email = True
                else:
                    logger.info("The email body is empty or contains only whitespace")
            else:
                logger.info("one or more keys in email dict is not Valid")
        else:
            logger.info("email does not have valid format")
                    
        if verified_email == True:
            logger.info(f"\nProcessing email {email['id']}...")    
            result = automation_system.process_email(email)    
            results.append(result)
        else:
            results.append = {"email_id": "N/A", "success":"N/A", "classification":"N/A", "response_sent":"N/A"}
    
    # Create a summary DataFrame
    df = pd.DataFrame(results)
    print("\nProcessing Summary:")
    print(df[["email_id", "success", "classification", "response_sent"]])

    return df


# Example usage:
if __name__ == "__main__":
    results_df = run_demonstration()