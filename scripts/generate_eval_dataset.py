import sys
import os
import json
import logging
import random
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.models import GenerationConfig
from src.retrieving.vector_store import ChromaDBManager
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are an expert AI dataset generator for a RAG evaluation pipeline.
Your task is to generate exactly 1 reading comprehension question based ONLY on the provided text chunk.

CRITICAL ANTI-BIAS RULES:
1. DO NOT copy the text verbatim, but you MAY use domain-specific keywords, function names, and technical terms found in the chunk. Paraphrase the surrounding context naturally.
2. The question must be realistic. Frame it as if a developer is confused and asking a question on StackOverflow or Discord.
3. The question MUST be answerable using only the provided text, but the question itself should look like it came from a human who hasn't read the text yet.
4. Categorize the difficulty as "easy", "medium", or "hard".
   - easy: Direct conceptual question.
   - medium: "How-to" or implementation question.
   - hard: Indirect, highly paraphrased, or scenario-based question.
5. Identify the primary "expected_topic" of the text (1-3 words).

Respond ONLY with a valid JSON object in this format:
{{
  "query": "The heavily paraphrased, realistic human question?",
  "expected_topic": "Topic Name",
  "difficulty": "medium"
}}

TEXT CHUNK:
{text}
"""


def _init_llm(config: GenerationConfig):
    provider = config.provider.lower()
    if provider == "gemini":
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
        client = genai.Client(api_key=api_key)

        def call_gemini(prompt: str) -> str:
            import time

            while True:
                try:
                    response = client.models.generate_content(
                        model=config.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.2),
                    )
                    # Force a 4-second delay to enforce the 15 RPM limit for Flash Lite
                    time.sleep(4)
                    return response.text
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        logger.warning(
                            "Gemini free tier rate limit hit. Waiting 45 seconds before retrying..."
                        )
                        time.sleep(45)
                    else:
                        raise e

        return call_gemini

    else:
        raise ValueError(f"Unknown provider: {provider}")


def clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate automated evaluation dataset"
    )
    parser.add_argument(
        "--num_queries", type=int, default=50, help="Number of queries to generate"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_datasets/auto_generated_eval.json",
    )
    args = parser.parse_args()

    config = GenerationConfig(provider="gemini", model_name="gemini-3.1-flash-lite")
    try:
        call_llm = _init_llm(config)
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return

    # Load chunks from ChromaDB
    collection_name = os.environ.get("CHROMA_COLLECTION", "unified_corpus")
    try:
        db_manager = ChromaDBManager(collection_name=collection_name)
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return

    all_docs = db_manager.collection.get()
    if not all_docs or not all_docs["ids"]:
        logger.error("No chunks found in ChromaDB. Run ingestion first.")
        return

    all_chunks = []
    for i in range(len(all_docs["ids"])):
        meta = all_docs["metadatas"][i]
        all_chunks.append(
            {
                "chunk_text": all_docs["documents"][i],
                "source_url": meta.get("source_url", meta.get("source_document", "")),
                "heading_path": (
                    meta.get("heading_path", "").split(" > ")
                    if meta.get("heading_path")
                    else []
                ),
            }
        )

    # Filter out very short chunks (need substance for a good question)
    valid_chunks = [
        c
        for c in all_chunks
        if c.get("word_count", len(c.get("chunk_text", "").split())) > 40
    ]

    if len(valid_chunks) < args.num_queries:
        logger.warning(
            f"Only found {len(valid_chunks)} valid chunks. Adjusting num_queries."
        )
        args.num_queries = len(valid_chunks)

    sampled_chunks = random.sample(valid_chunks, args.num_queries)

    generated_dataset = []

    logger.info(f"Generating {args.num_queries} evaluation queries...")
    for i, chunk in enumerate(sampled_chunks):
        text = chunk.get("chunk_text", "")
        prompt = PROMPT_TEMPLATE.format(text=text)

        logger.info(
            f"[{i+1}/{args.num_queries}] Generating question for chunk from {chunk.get('source_document')}"
        )

        try:
            response_text = call_llm(prompt)
            data = json.loads(clean_json(response_text))

            # Formulate the dataset entry matching our V3 eval format
            entry = {
                "query": data.get("query", ""),
                "expected_topic": data.get("expected_topic", "Unknown"),
                "expected_content_type": "concept",
                "acceptable_documents": [
                    chunk.get("source_url", chunk.get("source_document", ""))
                ],
                "acceptable_headings": chunk.get("heading_path", []),
                "difficulty": data.get("difficulty", "medium"),
                "category": "Auto-Generated",
            }
            generated_dataset.append(entry)

            # Save incrementally after every successful generation
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(generated_dataset, f, indent=2)

        except Exception as e:
            logger.error(
                f"Failed to generate/parse query for chunk {chunk.get('chunk_id')}: {e}"
            )
            continue

    logger.info(
        f"Successfully generated {len(generated_dataset)} queries. Saved to {args.output_file}"
    )


if __name__ == "__main__":
    main()
