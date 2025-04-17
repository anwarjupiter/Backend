import httpx
import asyncio
import time
import logging

# === Configure Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # logs to console
        logging.FileHandler("granite-3-3-8b-instruct.log", mode="a")  # logs to file
    ]
)

class AsyncAPITester:
    def __init__(self, url):
        self.test = url

    async def inject(self, method: str, **args):
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                files = args.pop("file", None)

                if method.lower() == "post":
                    response = await client.post(
                        url=self.test,
                        data=args,
                        files=files
                    )
                elif method.lower() == "get":
                    response = await client.get(url=self.test, params=args)
                else:
                    raise ValueError("Unsupported HTTP method.")

            elapsed = time.time() - start_time
            logging.info(f"[{method.upper()}] {self.test} -> {response.status_code} in {elapsed:.2f} seconds")
            return response

        except Exception as e:
            elapsed = time.time() - start_time
            logging.error(f"[{method.upper()}] {self.test} -> ERROR after {elapsed:.2f} seconds | {str(e)}")
            raise e


    
async def main():
    test1 = AsyncAPITester(url="http://127.0.0.1:8000/hello")
    response = await test1.inject(method="get")
    logging.info(response.text)

    test2 = AsyncAPITester(url="http://127.0.0.1:8000/agent")
    response = await test2.inject(method="post", question="Give me a random number")
    logging.info(response.text)

    test3 = AsyncAPITester(url="http://127.0.0.1:8000/agent")
    with open("input/tn.pdf", "rb") as f:
        files = {'file': ("tn.pdf", f, "application/pdf")}
        response = await test3.inject(
            method="post",
            question="What is the minimum road width of cottage industries?",
            file=files,
            vectorDB="ibm"
        )
        logging.info(response.text)

    test4 = AsyncAPITester(url="http://127.0.0.1:8000/agent")
    response = await test4.inject(
        method="post",
        question="How many customers are present in the dataset ?",
        mongo_uri="mongodb+srv://anwarmydheenk:xcwSgYCDarOKZzrq@cluster0.3t32wd8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        db_name="sample_analytics"
    )
    logging.info(response.text)

    test5 = AsyncAPITester(url="http://127.0.0.1:8000/agent")
    with open("input/civil.csv", "rb") as f:
        files = {'file': ("civil.csv", f, "text/csv")}
        response = await test5.inject(
            method="post", 
            question="How many rows do you have ?", 
            file=files
        )
        logging.info(response.text)
    
    test6 = AsyncAPITester(url="http://127.0.0.1:8000/agent")
    with open("input/tn.pdf", "rb") as f:
        files = {'file': ("tn.pdf", f, "application/pdf")}
        response = await test6.inject(
            method="post",
            question="What is the requirement of Ramp like gradient, width etc?",
            file=files,
            vectorDB="ibm"
        )
        logging.info(response.text)

    test7 = AsyncAPITester(url="http://127.0.0.1:8000/agent")
    response = await test7.inject(
        method="post",
        question="List the customer names who hold more than two accounts ",
        mongo_uri="mongodb+srv://anwarmydheenk:xcwSgYCDarOKZzrq@cluster0.3t32wd8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        db_name="sample_analytics"
    )
    logging.debug(response.text)

    test8 = AsyncAPITester(url="http://127.0.0.1:8000/agent")
    response = await test8.inject(
        method="post",
        question="List the top 10 tamil movies ",
        mongo_uri="mongodb+srv://anwarmydheenk:xcwSgYCDarOKZzrq@cluster0.3t32wd8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        db_name="sample_mflix"
    )
    logging.info(response.text)

    test9 = AsyncAPITester(url="http://127.0.0.1:8000/agent")
    with open("input/civil.csv", "rb") as f:
        files = {'file': ("civil.csv", f, "text/csv")}
        response = await test9.inject(
            method="post", 
            question="Which equipments does higher energy consumption compared with others ?", 
            file=files
        )
        logging.info(response.text)
    
    test10 = AsyncAPITester(url="http://127.0.0.1:8000/agent")
    with open("input/tn.pdf", "rb") as f:
        files = {'file': ("tn.pdf", f, "application/pdf")}
        response = await test10.inject(
            method="post",
            question="What is the maximum permissible FSI for IT Building in City area ?",
            file=files,
            vectorDB="ibm"
        )
        logging.info(response.text)

# Run the async test suite
asyncio.run(main())