from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """\
You are a structured data extractor for company intelligence.

Extract company profile facts from the provided web page content.
Only extract facts explicitly stated in the content — do not infer or hallucinate.
Leave fields as null if the information is not present in the content.

Extract:
- name: Official company name
- description: What the company does (2-3 sentences max)
- industry: Primary industry or sector (e.g. "E-commerce", "Cloud Computing", "FinTech")
- headquarters: City, State/Country of main office (e.g. "San Jose, California")
- founded_year: Year the company was founded (integer only, e.g. 1995)
- employee_count: Number of employees, can be approximate (e.g. "~50,000", "10,000+")
- business_model: How the company makes money (e.g. "B2B SaaS", "marketplace", "B2C subscription")
- products: List of main products or services (concise names only)
- target_customers: List of customer segments (e.g. ["consumers", "enterprise", "SMBs"])
- markets: List of geographic or vertical markets served (e.g. ["North America", "Europe"])
"""

EXTRACTION_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("human", "Company: {company_name}\n\nPage content:\n{content}"),
    ]
)
