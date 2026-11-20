import os
from typing import Literal, Optional
from pydantic import BaseModel, Field
from atlassian import Jira
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1. Initialize the Jira client
jira_client = Jira(
    url=os.environ["JIRA_INSTANCE_URL"],
    username=os.environ["JIRA_USERNAME"],
    password=os.environ["JIRA_API_TOKEN"],
    cloud=True
)

# 2. Define the input schema for strict LLM validation
class JiraTicketInput(BaseModel):
    project_key: str = Field(
        default="KAN",
        description="The uppercase key of the Jira project (e.g., 'KAN', 'ENG', 'SUP', 'WEB')."
    )
    summary: str = Field(
        description="A concise, single-line title summarizing the issue."
    )
    description: str = Field(
        description="Detailed steps to reproduce, context, or acceptance criteria."
    )
    issue_type: Literal["Bug", "Task", "Story"] = Field(
        default="Task",
        description="The type of Jira ticket to create."
    )
    priority: Optional[Literal["Highest", "High", "Medium", "Low", "Lowest"]] = Field(
        default="Medium",
        description="The priority level of the issue."
    )

# 3. Create the LangChain Tool
@tool(args_schema=JiraTicketInput)
def create_jira_ticket(
    project_key: str,
    summary: str,
    description: str,
    issue_type: str = "Task",
    priority: str = "Medium"
) -> str:
    """Creates a new ticket in Jira and returns the issue key and URL."""
    fields = {
        "project": {"key": project_key},
        "summary": summary,
        "description": description,
        "issuetype": {"name": issue_type},
        "priority": {"name": priority},
    }
    
    try:
        response = jira_client.create_issue(fields=fields)
        issue_key = response.get("key")
        issue_url = f"{os.environ['JIRA_INSTANCE_URL']}/browse/{issue_key}"
        return f"Successfully created Jira ticket {issue_key}: {issue_url}"
    except Exception as e:
        return f"Failed to create Jira ticket: {str(e)}"


@tool
def search_jira_tickets(jql_query: str, max_results: int = 10) -> list[dict]:
    """
    Search and retrieve Jira tickets using a JQL (Jira Query Language) query.
    
    Args:
        jql_query: A valid JQL string (e.g., 'project = "ENG" AND status = "In Progress"').
        max_results: Maximum number of tickets to return (default is 10).
    """
    jira = jira_client
    
    # Only request necessary fields to keep token usage low
    fields = ["summary", "status", "assignee", "priority", "description", "updated"]
    response = jira.jql(jql_query, limit=max_results, fields=fields)
    
    tickets = []
    for issue in response.get("issues", []):
        issue_fields = issue.get("fields", {})
        assignee = issue_fields.get("assignee")
        
        tickets.append({
            "key": issue.get("key"),
            "summary": issue_fields.get("summary"),
            "status": issue_fields.get("status", {}).get("name"),
            "priority": issue_fields.get("priority", {}).get("name"),
            "assignee": assignee.get("displayName") if assignee else "Unassigned",
            "updated": issue_fields.get("updated"),
            "description": issue_fields.get("description"),
        })
        
    return tickets

jira_tools = [create_jira_ticket, search_jira_tickets]
