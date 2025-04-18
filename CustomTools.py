import random
from langchain_core.tools import tool
from bundle.PdfTeacher import PDFQABot
from bundle.PandasDoctor import PandasDoctor
from bundle.MongoTool import MongoAggregationTool


@tool
def pdf_tool(file: str, vectorDB: str, question: str) -> str:
    """Provide a valid answer from the given PDF.
    Args:
        file: The file path of the uploaded PDF.
        vectorDB: The path to the vector store.
        question: The user's question about the document.
    """
    pdfBot = PDFQABot()
    pdfBot._build_qa_chain(pdf_path=file, vectorDB=vectorDB)
    answer = pdfBot.ask(question=question)
    return answer


@tool 
def csv_tool(file:str,question:str)->str:
    """
    Provides a exact answer or data from the given CSV file.
    Args:
        file:The file path of the uploaded CSV.
        question: The user's question about the given data.
    """
    pandasBot = PandasDoctor()
    pandasBot._load_dataset(path=file)
    answer = pandasBot.run(question=question)
    return answer

@tool
def mongo_tool(mongo_uri:str,db_name:str,question:str)->str:
    """
    Provides a exact answer or data from MongoDB Database
    Args:
        mongo_uri: The Mongo URI to connect the client
        db_name: The database name of the MongoDB
        question: The user's question into the database
    """
    mongoBot = MongoAggregationTool(connection_string=mongo_uri,db_name=db_name)
    answer = mongoBot.run(query=question)
    return answer

@tool
def get_joke(category: str = "any", count: int = 1) -> str:
    """Tells one or more jokes from a specific category. 
    Args:
        category: Type of joke like 'tech', 'dad', 'animal', or 'any'.
        count: Number of jokes to return (default 1).
    """
    jokes = {
        "tech": [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "There are only 10 kinds of people in this world: those who understand binary and those who don’t."
        ],
        "dad": [
            "I'm reading a book on anti-gravity. It's impossible to put down!",
            "I would avoid the sushi if I was you. It’s a little fishy."
        ],
        "any": [
            "Why did the chicken cross the road? To get to the other side!",
            "Parallel lines have so much in common. It’s a shame they’ll never meet."
        ]
    }
    chosen = jokes.get(category.lower(), jokes["any"])
    return "\n".join(random.choices(chosen, k=min(count, len(chosen))))

@tool
def get_weather(location: str, unit: str = "Celsius") -> str:
    """Provides weather information for a given location.
    Args:
        location: Name of the city or region.
        unit: Unit of temperature. Either 'Celsius' or 'Fahrenheit'.
    """
    temp = 25 if unit == "Celsius" else 77
    return f"The weather in {location} is sunny and {temp}°{unit[0]}."

@tool
def get_quote(topic: str = "life", author: str = None) -> str:
    """Generates an inspirational quote based on topic or author.
    Args:
        topic: Theme like 'life', 'success', 'love'.
        author: Optional author name for a quote.
    """
    quotes = {
        "life": [
            "Life is what happens when you're busy making other plans.",
            "The purpose of our lives is to be happy."
        ],
        "success": [
            "Success usually comes to those who are too busy to be looking for it.",
            "Don't be afraid to give up the good to go for the great."
        ]
    }
    quote = random.choice(quotes.get(topic.lower(), quotes["life"]))
    if author:
        return f"{quote} - {author}"
    return quote

@tool
def uruttu():
    """Generates an random number between 1 and 6.
    """
    return random.randint(1,6)