import os
import json
import time
import random
import litellm
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import argparse

# Load LitSearch dataset from HuggingFace
def load_litsearch_dataset():
    try:
        print("Loading LitSearch dataset...")
        query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
        corpus_data = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
        
        # Print some basic info
        print(f"Loaded {len(query_data)} queries and {len(corpus_data)} corpus documents")
        
        # Debug: Print keys from first items
        if len(query_data) > 0:
            print("Query keys:", list(query_data[0].keys()))
        if len(corpus_data) > 0:
            print("Corpus keys:", list(corpus_data[0].keys()))
        
        return query_data, corpus_data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have the 'datasets' library installed: pip install datasets")
        exit(1)

# Define a prompt to transform paper-seeking queries into conceptual questions
def create_transformation_prompt(query, paper_title, paper_abstract, paper_full_text=None):
    # Truncate full text if it's too long to fit in context window
    # GPT-4o has ~8K token limit, keeping prompt under ~4K tokens for response space
    max_full_text_length = 32000  # Characters, roughly ~2K tokens
    
    paper_content = paper_abstract
    if paper_full_text and len(paper_full_text.strip()) > 0:
        # Use both abstract and truncated full text
        truncated_full_text = paper_full_text[:max_full_text_length] + "..." if len(paper_full_text) > max_full_text_length else paper_full_text
        paper_content = f"Abstract:\n{paper_abstract}\n\nFull Paper (may be truncated):\n{truncated_full_text}"
    
    system_prompt = """You are an expert academic assistant. Your task is to:
1. Take a paper-seeking query and a scientific paper (title + abstract + full text if available)
2. Create a new conceptual question that asks about the information/findings in the paper rather than asking for the paper itself
3. Generate a comprehensive ground truth answer based on the paper's content.

The new question should NOT ask for paper recommendations but instead ask about concepts, methods, findings, or implications discussed in the paper.

Be specific and detailed in your answer, drawing from the full paper text when available to provide a thorough response."""

    user_prompt = f"""
Original Query: {query}

Paper Title: {paper_title}

Paper Content:
{paper_content}

Please generate:
1. A natural conceptual question that someone might ask when seeking to understand the information in this paper (rather than asking for paper recommendations)
2. A comprehensive ground truth answer to the conceptual question that you generate based on the paper's content. The answer should include the paper's title, and should be under 100 words.

Format your response as a JSON object with two fields:
- conceptual_question: The new question
- ground_truth_answer: The answer to the question based on the paper
"""
    return system_prompt, user_prompt

# Function to call GPT-4o via LiteLLM
def transform_query_with_gpt(query, paper_title, paper_abstract, paper_full_text, api_key, base_url, model_name):
    system_prompt, user_prompt = create_transformation_prompt(query, paper_title, paper_abstract, paper_full_text)
    
    try:
        response = litellm.completion(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1024
        )
        
        # Parse JSON response
        content = response.choices[0].message.content
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response, attempting fallback extraction...")
            # Fallback extraction for non-JSON responses
            lines = content.split('\n')
            conceptual_question = ""
            ground_truth_answer = ""
            
            in_question = False
            in_answer = False
            
            for line in lines:
                if "conceptual_question" in line.lower() or "conceptual question" in line.lower():
                    in_question = True
                    in_answer = False
                    # Try to extract from this line first
                    if ":" in line:
                        conceptual_question = line.split(":", 1)[1].strip()
                    continue
                
                if "ground_truth_answer" in line.lower() or "ground truth answer" in line.lower():
                    in_question = False
                    in_answer = True
                    # Try to extract from this line first
                    if ":" in line:
                        ground_truth_answer = line.split(":", 1)[1].strip()
                    continue
                
                # Accumulate text for multi-line responses
                if in_question and not line.strip().startswith("ground_truth") and not line.strip().startswith("Ground truth"):
                    conceptual_question += " " + line.strip()
                
                if in_answer:
                    ground_truth_answer += " " + line.strip()
            
            # Clean up results
            conceptual_question = conceptual_question.strip()
            ground_truth_answer = ground_truth_answer.strip()
            
            if not conceptual_question or not ground_truth_answer:
                print("Could not extract question and answer from response")
                return None
                
            return {
                "conceptual_question": conceptual_question,
                "ground_truth_answer": ground_truth_answer
            }
    except Exception as e:
        print(f"Error calling GPT: {e}")
        return None

# Build a lookup dictionary for faster paper retrieval
def build_corpus_lookup(corpus_data):
    print("Building corpus lookup dictionary...")
    lookup = {}
    paper_id_field = None
    
    # First, determine which field contains the paper ID
    if len(corpus_data) > 0:
        sample = corpus_data[0]
        if "paper_id" in sample:
            paper_id_field = "paper_id"
        elif "doc_id" in sample:
            paper_id_field = "doc_id"
        elif "id" in sample:
            paper_id_field = "id"
        else:
            # Try to guess the field by looking for ID-like fields
            for key in sample.keys():
                if "id" in key.lower():
                    paper_id_field = key
                    break
    
    if not paper_id_field:
        print("WARNING: Could not determine paper ID field in corpus. Using first field as fallback.")
        paper_id_field = list(corpus_data[0].keys())[0] if len(corpus_data) > 0 else None
    
    print(f"Using '{paper_id_field}' as the paper ID field")
    
    # Build the lookup dictionary
    if paper_id_field:
        for item in tqdm(corpus_data):
            paper_id = item.get(paper_id_field)
            if paper_id:
                lookup[paper_id] = item
    
    print(f"Built lookup with {len(lookup)} papers")
    return lookup, paper_id_field

# Main function to generate the synthetic dataset
def generate_synthetic_dataset(num_samples=500, output_file="litsearch_rag_dataset.json", 
                              api_key=None, base_url="https://cmu.litellm.ai", 
                              model_name="openai/gpt-4o"):
    # Get API key from environment variable if not provided
    if not api_key:
        api_key = os.environ.get("LITELLM_API_KEY")
        if not api_key:
            raise ValueError("Please set the LITELLM_API_KEY environment variable or provide it as an argument")
    
    # Load the LitSearch dataset
    query_data, corpus_data = load_litsearch_dataset()
    
    # Build a lookup dictionary for faster paper retrieval
    corpus_lookup, paper_id_field = build_corpus_lookup(corpus_data)
    
    # Create a list to store our synthetic data
    synthetic_data = []
    processed_data = []
    
    # Determine gold paper ID field in query data
    gold_paper_id_field = None
    if len(query_data) > 0:
        sample = query_data[0]
        possible_fields = ["gold_paper_ids", "gold_document_ids", "relevant_document_ids", "corpusids"]
        for field in possible_fields:
            if field in sample and sample[field]:
                gold_paper_id_field = field
                break
    
    if not gold_paper_id_field:
        print("ERROR: Could not find gold paper ID field in query data")
        # Print sample keys for debugging
        if len(query_data) > 0:
            print(f"Available fields in query_data: {list(query_data[0].keys())}")
            print(f"Sample of first query: {query_data[0]}")
        return []
    
    print(f"Using '{gold_paper_id_field}' as the gold paper ID field")
    
    print("Processing queries and matching with relevant papers...")
    for query_index, query_item in enumerate(tqdm(query_data)):
        # Get the query
        original_query = query_item.get("query", "")
        if not original_query:
            continue
            
        # Get relevant paper IDs
        gold_paper_ids = query_item.get(gold_paper_id_field, [])
        if not gold_paper_ids:
            continue
        
        matched = 0
        # Match with papers in corpus
        for paper_id in gold_paper_ids:
            # Use lookup for faster retrieval
            paper = corpus_lookup.get(paper_id)
            
            if paper:
                paper_title = paper.get("title", "")
                paper_abstract = paper.get("abstract", "")
                paper_full_text = paper.get("full_paper", "")  # Get full paper text
                
                # Skip if we don't have either title or abstract
                if not paper_title or not paper_abstract:
                    continue
                
                # Add to processed data
                processed_data.append({
                    "original_query": original_query,
                    "paper_title": paper_title,
                    "paper_abstract": paper_abstract,
                    "paper_full_text": paper_full_text,  # Store full paper text
                    "paper_id": paper_id
                })
                matched += 1
        
        if matched > 0 and query_index < 5:  # Only print for first few queries
            print(f"Query: '{original_query[:50]}...' matched with {matched} papers")
    
    print(f"Found {len(processed_data)} matching query-paper pairs")
    
    # Randomly sample if we have more than we need
    if len(processed_data) > num_samples:
        print(f"Randomly sampling {num_samples} pairs from {len(processed_data)} available pairs")
        processed_data = random.sample(processed_data, num_samples)
    
    # Process each sample
    print(f"Generating {len(processed_data)} synthetic query-answer pairs...")
    for idx, sample in enumerate(tqdm(processed_data)):
        try:
            # Extract data from the processed sample
            original_query = sample["original_query"]
            paper_title = sample["paper_title"]
            paper_abstract = sample["paper_abstract"]
            paper_full_text = sample.get("paper_full_text", "")  # Get full paper text, default to empty string
            paper_id = sample["paper_id"]
            
            # Transform the query using GPT-4o
            transformed = transform_query_with_gpt(original_query, paper_title, paper_abstract, paper_full_text,
                                                api_key, base_url, model_name)
            
            if transformed:
                # Add original data as well for reference
                entry = {
                    "original_query": original_query,
                    "paper_title": paper_title,
                    "paper_abstract": paper_abstract,
                    # Don't store the full paper text in the final dataset to keep it manageable
                    # Just include a flag indicating if full text was used in generation
                    "used_full_text": bool(paper_full_text and len(paper_full_text.strip()) > 0),
                    "paper_id": paper_id,
                    "conceptual_question": transformed["conceptual_question"],
                    "ground_truth_answer": transformed["ground_truth_answer"]
                }
                synthetic_data.append(entry)
                
                # Print progress and a sample every 10 entries
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(processed_data)} samples")
                    if idx < 3:  # Show sample for first few conversions
                        print(f"Original: {original_query}")
                        print(f"Transformed: {transformed['conceptual_question']}")
                        print(f"Answer start: {transformed['ground_truth_answer'][:100]}...")
                    
                # Sleep to avoid hitting API rate limits
                time.sleep(1)
                
                # Stop if we've reached the desired number of samples
                if len(synthetic_data) >= num_samples:
                    break
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Save the dataset to a JSON file
    print(f"Saving {len(synthetic_data)} synthetic samples to {output_file}")
    with open(output_file, 'w') as f:
        json.dump({"data": synthetic_data}, f, indent=2)
    
    # Also save as CSV for easier viewing
    df = pd.DataFrame(synthetic_data)
    csv_file = output_file.replace('.json', '.csv')
    df.to_csv(csv_file, index=False)
    print(f"Also saved data to {csv_file}")
    
    return synthetic_data

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Generate a synthetic RAG dataset from LitSearch")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--output_file", type=str, default="litsearch_rag_dataset_fullpaper_500.json", help="Output file path")
    parser.add_argument("--api_key", type=str, help="LiteLLM API key (if not set in environment)")
    parser.add_argument("--base_url", type=str, default="https://cmu.litellm.ai", help="LiteLLM base URL")
    parser.add_argument("--model_name", type=str, default="openai/gpt-4o", help="Model name to use")
    args = parser.parse_args()
    
    # Generate synthetic dataset
    synthetic_data = generate_synthetic_dataset(
        num_samples=args.num_samples,
        output_file=args.output_file,
        api_key="Replace with your API key",
        base_url=args.base_url,
        model_name=args.model_name
    )
    
    # Print a sample for inspection
    if synthetic_data:
        print("\nSample transformed data:")
        sample = random.choice(synthetic_data)
        print(f"Original Query: {sample['original_query']}")
        print(f"Paper Title: {sample['paper_title']}")
        print(f"New Conceptual Question: {sample['conceptual_question']}")
        print(f"Ground Truth Answer (first 100 chars): {sample['ground_truth_answer'][:100]}...")
    else:
        print("\nNo data was generated. Please check the error messages above.")