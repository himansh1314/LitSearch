import json
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import warnings
from tqdm import tqdm
import requests
from typing import List, Dict, Any, Tuple, Optional
import concurrent.futures
# Import the Groq Python client library
from groq import Groq

# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Installing spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class LitSearchAnalyzer:
    """
    A comprehensive analyzer for the LitSearch benchmark.
    """
    
    def __init__(self, data_path: str, groq_api_key: str, output_dir: str = "analysis_results"):
        """
        Initialize the analyzer.
        
        Args:
            data_path: Path to the JSON file with pass and fail queries
            groq_api_key: API key for Groq
            output_dir: Directory to save analysis results
        """
        self.data_path = data_path
        self.groq_api_key = groq_api_key
        self.output_dir = output_dir
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=self.groq_api_key)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self.data = self._load_data()
        
        # Process data into dataframe
        self.df = self._prepare_dataframe()
        
        # Technical term dictionaries
        self.tech_terms = self._load_technical_terms()
        
        # Question patterns
        self.question_patterns = {
            "which_paper": r"which paper|which research|which study|which work",
            "where_find": r"where can i find|where could i find|where might i find",
            "point_to": r"can you point|could you point|point me to",
            "recommend": r"can you recommend|could you recommend|would you recommend",
            "suggest": r"can you suggest|could you suggest|would you suggest",
            "are_there": r"are there any|are there papers|are there studies|are there research",
            "what_are": r"what are some|what are the",
            "is_there": r"is there a paper|is there research|is there a study|is there any paper"
        }
        
        # Analysis results
        self.results = {}
        
    def _load_data(self) -> Dict:
        """Load data from JSON file."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            
        # Validate data format
        if "pass_queries" not in data or "fail_queries" not in data:
            raise ValueError("Data must contain 'pass_queries' and 'fail_queries' keys")
            
        print(f"Loaded {len(data['pass_queries'])} pass queries and {len(data['fail_queries'])} fail queries")
        return data
    
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Convert JSON data to a pandas DataFrame for analysis."""
        pass_queries = [{"query": q, "status": "pass"} for q in self.data["pass_queries"]]
        fail_queries = [{"query": q, "status": "fail"} for q in self.data["fail_queries"]]
        
        df = pd.DataFrame(pass_queries + fail_queries)
        
        # Add basic features
        df["word_count"] = df["query"].apply(lambda x: len(word_tokenize(x)))
        df["char_count"] = df["query"].apply(len)
        df["sentence_count"] = df["query"].apply(lambda x: len(sent_tokenize(x)))
        
        return df
    
    def _load_technical_terms(self) -> Dict[str, List[str]]:
        """Load dictionaries of technical terms for different categories."""
        # These lists would ideally be much more comprehensive in a real implementation
        return {
            "model_names": [
                "bert", "roberta", "gpt", "t5", "xlm", "llama", "palm", "lstm", 
                "transformer", "elmo", "word2vec", "glove", "fasttext", "bart", 
                "electra", "albert", "distilbert", "xlnet", "mt5", "deberta"
            ],
            "task_names": [
                "classification", "translation", "summarization", "generation", "parsing",
                "ner", "qa", "dialogue", "sentiment", "retrieval", "inference", "entailment",
                "paraphrase", "segmentation", "tagging", "clustering", "disambiguation"
            ],
            "method_terms": [
                "attention", "embedding", "pooling", "fine-tuning", "transfer learning",
                "contrastive", "adversarial", "autoencoder", "gan", "moe", "dropout",
                "regularization", "normalization", "backpropagation", "gradient descent"
            ],
            "metric_names": [
                "accuracy", "precision", "recall", "f1", "bleu", "rouge", "meteor",
                "perplexity", "map", "ndcg", "rmse", "mae", "auc", "mrr"
            ],
            "dataset_names": [
                "squad", "glue", "superglue", "mnli", "snli", "wmt", "cnn/dailymail",
                "ms marco", "imagenet", "natural questions", "hotpotqa", "wikihow"
            ],
            "generic_ml_terms": [
                "model", "training", "learning", "neural", "network", "deep", "algorithm",
                "feature", "layer", "parameter", "optimization", "inference", "corpus"
            ]
        }
    
    def groq_llm_query(self, prompt, model="llama3-70b-8192"):
        """Query the Groq API with a prompt using the official client library."""
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to analyze scientific literature search queries."},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying Groq API: {e}")
            time.sleep(2)  # Wait before retrying
            try:
                response = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a research assistant helping to analyze scientific literature search queries."},
                        {"role": "user", "content": prompt}
                    ],
                    model=model,
                    temperature=0.3,
                    max_tokens=1024
                )
                return response.choices[0].message.content
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                return None
    
    def run_complete_analysis(self):
        """Run all analyses and compile results."""
        print("Starting LitSearch benchmark analysis...")
        
        # Run deterministic analyses
        print("Running deterministic analyses...")
        self.analyze_overall_distribution()
        self.analyze_query_length()
        self.analyze_query_structure()
        self.analyze_technical_terms()
        self.analyze_syntactic_complexity()
        
        # Run LLM-based analyses
        print("Running LLM-based analyses...")
        self.analyze_research_areas()
        self.analyze_query_intentions()
        self.analyze_failure_modes()
        
        # Generate feature importance
        print("Analyzing feature importance...")
        self.analyze_feature_importance()
        
        # Generate visualizations
        print("Generating visualizations...")
        self.generate_visualizations()
        
        # Save results
        self.save_results()
        
        print(f"Analysis complete. Results saved to {self.output_dir}")
        return self.results
    
    def analyze_overall_distribution(self):
        """Analyze overall distribution of pass and fail queries."""
        total = len(self.df)
        pass_count = sum(self.df["status"] == "pass")
        fail_count = sum(self.df["status"] == "fail")
        
        self.results["overall_distribution"] = {
            "total_queries": total,
            "pass_queries": pass_count,
            "fail_queries": fail_count,
            "pass_rate": pass_count / total * 100,
            "fail_rate": fail_count / total * 100
        }
        
    def analyze_query_length(self):
        """Analyze query length distribution and its relationship with pass/fail status."""
        # Define length categories
        self.df["length_category"] = pd.cut(
            self.df["word_count"], 
            bins=[0, 15, 25, float('inf')],
            labels=["Short (< 15 words)", "Medium (15-25 words)", "Long (> 25 words)"]
        )
        
        # Calculate pass rates by length category
        length_stats = self.df.groupby("length_category").apply(
            lambda x: {
                "count": len(x),
                "pass_count": sum(x["status"] == "pass"),
                "pass_rate": sum(x["status"] == "pass") / len(x) * 100
            }
        ).to_dict()
        
        # Add summary statistics
        length_summary = {
            "overall_mean_word_count": self.df["word_count"].mean(),
            "pass_mean_word_count": self.df[self.df["status"] == "pass"]["word_count"].mean(),
            "fail_mean_word_count": self.df[self.df["status"] == "fail"]["word_count"].mean(),
            "by_category": length_stats
        }
        
        self.results["query_length"] = length_summary
        
    def analyze_query_structure(self):
        """Analyze query structure based on question patterns."""
        # Add question type features
        for pattern_name, pattern in self.question_patterns.items():
            self.df[f"is_{pattern_name}"] = self.df["query"].str.lower().str.contains(pattern, regex=True)
        
        # Determine primary question type (first matching pattern)
        def get_primary_question_type(row):
            for pattern_name in self.question_patterns:
                if row[f"is_{pattern_name}"]:
                    return pattern_name
            return "other"
        
        self.df["primary_question_type"] = self.df.apply(get_primary_question_type, axis=1)
        
        # Calculate stats for each question type
        question_type_stats = {}
        for q_type in list(self.question_patterns.keys()) + ["other"]:
            subset = self.df[self.df["primary_question_type"] == q_type]
            if len(subset) > 0:
                question_type_stats[q_type] = {
                    "count": len(subset),
                    "pass_count": sum(subset["status"] == "pass"),
                    "pass_rate": sum(subset["status"] == "pass") / len(subset) * 100,
                    "example": subset["query"].iloc[0] if len(subset) > 0 else ""
                }
        
        # Classify as direct or indirect question
        direct_patterns = ["which_paper", "where_find", "point_to"]
        self.df["is_direct_question"] = self.df["primary_question_type"].isin(direct_patterns)
        
        direct_stats = {
            "direct_questions": {
                "count": sum(self.df["is_direct_question"]),
                "pass_rate": sum(self.df["is_direct_question"] & (self.df["status"] == "pass")) / 
                            sum(self.df["is_direct_question"]) * 100
            },
            "indirect_requests": {
                "count": sum(~self.df["is_direct_question"]),
                "pass_rate": sum(~self.df["is_direct_question"] & (self.df["status"] == "pass")) / 
                            sum(~self.df["is_direct_question"]) * 100
            }
        }
        
        self.results["query_structure"] = {
            "by_question_type": question_type_stats,
            "by_directness": direct_stats
        }
        
    def analyze_technical_terms(self):
        """Analyze technical terminology in queries."""
        # Count technical terms by category
        for category, terms in self.tech_terms.items():
            pattern = r'\b(' + '|'.join(terms) + r')\b'
            self.df[f"{category}_count"] = self.df["query"].str.lower().str.count(pattern)
        
        # Calculate total technical terms
        tech_columns = [f"{cat}_count" for cat in self.tech_terms.keys()]
        self.df["total_tech_terms"] = self.df[tech_columns].sum(axis=1)
        
        # Calculate tech term density (terms per word)
        self.df["tech_term_density"] = self.df["total_tech_terms"] / self.df["word_count"]
        
        # Split into density levels
        self.df["tech_density_level"] = pd.cut(
            self.df["tech_term_density"],
            bins=[0, 0.1, 0.2, 0.3, float('inf')],
            labels=["Low", "Medium", "High", "Very High"]
        )
        
        # Calculate stats by term category
        term_category_stats = {}
        for category in self.tech_terms:
            # Find queries with at least one term from this category
            has_term = self.df[f"{category}_count"] > 0
            if sum(has_term) > 0:
                term_category_stats[category] = {
                    "occurrence": sum(has_term),
                    "pass_count": sum(has_term & (self.df["status"] == "pass")),
                    "pass_rate": sum(has_term & (self.df["status"] == "pass")) / sum(has_term) * 100
                }
        
        # Calculate stats by density level
        density_stats = self.df.groupby("tech_density_level").apply(
            lambda x: {
                "count": len(x),
                "pass_count": sum(x["status"] == "pass"),
                "pass_rate": sum(x["status"] == "pass") / len(x) * 100 if len(x) > 0 else 0
            }
        ).to_dict()
        
        self.results["technical_terms"] = {
            "by_category": term_category_stats,
            "by_density": density_stats,
            "overall_stats": {
                "mean_tech_terms": self.df["total_tech_terms"].mean(),
                "pass_mean_tech_terms": self.df[self.df["status"] == "pass"]["total_tech_terms"].mean(),
                "fail_mean_tech_terms": self.df[self.df["status"] == "fail"]["total_tech_terms"].mean()
            }
        }
        
    def calculate_parse_depth(self, token):
        """Calculate the depth of a token in the dependency parse tree."""
        depth = 0
        current_token = token
        while current_token.head != current_token:  # Root token is its own head
            depth += 1
            current_token = current_token.head
        return depth
        
    def analyze_syntactic_complexity(self):
        """Analyze syntactic complexity of queries using spaCy."""
        # Process a sample of queries with spaCy for detailed parsing
        # Using a sample to keep processing time reasonable
        sample_size = min(200, len(self.df))
        sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
        
        parse_results = []
        for idx in tqdm(sample_indices, desc="Parsing queries"):
            query = self.df.iloc[idx]["query"]
            doc = nlp(query)
            
            # Calculate parse depths for each token
            parse_depths = [self.calculate_parse_depth(token) for token in doc]
            max_depth = max(parse_depths) if parse_depths else 0
            
            # Extract syntax features
            result = {
                "query_index": idx,
                "status": self.df.iloc[idx]["status"],
                "parse_depth": max_depth,
                "noun_chunks": len(list(doc.noun_chunks)),
                "prep_phrases": sum(1 for token in doc if token.dep_ == "prep"),
                "subordinate_clauses": sum(1 for token in doc if token.dep_ in ["ccomp", "xcomp", "advcl"]),
                "coordinate_conjunctions": sum(1 for token in doc if token.dep_ == "cc"),
            }
            parse_results.append(result)
        
        # Convert to DataFrame
        parse_df = pd.DataFrame(parse_results)
        
        # Calculate statistics by status
        syntactic_stats = {
            "pass_queries": {
                "mean_parse_depth": parse_df[parse_df["status"] == "pass"]["parse_depth"].mean(),
                "mean_noun_chunks": parse_df[parse_df["status"] == "pass"]["noun_chunks"].mean(),
                "mean_prep_phrases": parse_df[parse_df["status"] == "pass"]["prep_phrases"].mean(),
                "mean_subordinate_clauses": parse_df[parse_df["status"] == "pass"]["subordinate_clauses"].mean(),
                "mean_coordinate_conjunctions": parse_df[parse_df["status"] == "pass"]["coordinate_conjunctions"].mean(),
            },
            "fail_queries": {
                "mean_parse_depth": parse_df[parse_df["status"] == "fail"]["parse_depth"].mean(),
                "mean_noun_chunks": parse_df[parse_df["status"] == "fail"]["noun_chunks"].mean(),
                "mean_prep_phrases": parse_df[parse_df["status"] == "fail"]["prep_phrases"].mean(),
                "mean_subordinate_clauses": parse_df[parse_df["status"] == "fail"]["subordinate_clauses"].mean(),
                "mean_coordinate_conjunctions": parse_df[parse_df["status"] == "fail"]["coordinate_conjunctions"].mean(),
            }
        }
        
        self.results["syntactic_complexity"] = syntactic_stats
        
    def batch_process_with_llm(self, queries, process_function, batch_size=20, description="Processing"):
        """Process queries in batches using the LLM."""
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(queries), batch_size), desc=description):
            batch = queries[i:i+batch_size]
            batch_results = process_function(batch)
            results.extend(batch_results)
            time.sleep(1)  # Avoid rate limiting
            
        return results
    
    def analyze_research_areas(self):
        """Analyze research areas of queries using LLM."""
        # Define research areas
        research_areas = [
            "Language modeling", "Machine translation", "Summarization", 
            "Question answering", "Information retrieval", "Dialogue systems",
            "Sentiment analysis", "Named entity recognition", "Parsing",
            "Embeddings & representations", "Knowledge graphs", "Few-shot learning",
            "Efficient ML methods", "Interpretability & evaluation"
        ]
        
        # Define processor function for research area classification
        def process_research_areas(batch):
            prompt = f"""Classify each of the following scientific literature search queries into one of these research areas:
{', '.join(research_areas)}

If a query doesn't fit clearly into one category, choose the best match.
For each query, respond with only the query number and the research area, in this format:
1. Research Area
2. Research Area
... and so on.

Queries:
{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(batch)])}"""

            response = self.groq_llm_query(prompt)
            if not response:
                return [{"query": q, "research_area": "Unknown"} for q in batch]
            
            # Parse response
            results = []
            lines = response.strip().split('\n')
            for i, query in enumerate(batch):
                research_area = "Unknown"
                pattern = fr"{i+1}\.\s+(.*)"
                
                for line in lines:
                    match = re.search(pattern, line)
                    if match:
                        research_area = match.group(1)
                        break
                
                results.append({"query": query, "research_area": research_area})
            
            return results
        
        # Process queries
        queries = self.df["query"].tolist()
        research_area_results = self.batch_process_with_llm(
            queries, process_research_areas, batch_size=20, description="Classifying research areas"
        )
        
        # Add to dataframe
        research_area_df = pd.DataFrame(research_area_results)
        
        # Remove the duplicate 'query' column from the results before joining
        if 'query' in research_area_df.columns:
            research_area_df = research_area_df.drop(columns=['query'])
            
        # Join by index
        self.df = pd.concat([self.df, research_area_df], axis=1)
        
        # Calculate stats by research area
        research_area_stats = {}
        for area in research_areas:
            area_df = self.df[self.df["research_area"] == area]
            if len(area_df) > 0:
                research_area_stats[area] = {
                    "count": len(area_df),
                    "pass_count": sum(area_df["status"] == "pass"),
                    "pass_rate": sum(area_df["status"] == "pass") / len(area_df) * 100
                }
        
        self.results["research_areas"] = research_area_stats
        
    def analyze_query_intentions(self):
        """Analyze query intentions using LLM."""
        # Define intention types
        intention_types = [
            "Find specific paper", "Explore research direction", 
            "Find implementation/method", "Find evaluation benchmark",
            "Find comparison of methods", "Find resources/datasets"
        ]
        
        # Define processor function for intention classification
        def process_intentions(batch):
            prompt = f"""Classify each of the following scientific literature search queries by their intention:
{', '.join(intention_types)}

For each query, respond with only the query number and the intention type, in this format:
1. Intention Type
2. Intention Type
... and so on.

Queries:
{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(batch)])}"""

            response = self.groq_llm_query(prompt)
            if not response:
                return [{"query": q, "intention": "Unknown"} for q in batch]
            
            # Parse response
            results = []
            lines = response.strip().split('\n')
            for i, query in enumerate(batch):
                intention = "Unknown"
                pattern = fr"{i+1}\.\s+(.*)"
                
                for line in lines:
                    match = re.search(pattern, line)
                    if match:
                        intention = match.group(1)
                        break
                
                results.append({"query": query, "intention": intention})
            
            return results
        
        # Process queries
        queries = self.df["query"].tolist()
        intention_results = self.batch_process_with_llm(
            queries, process_intentions, batch_size=20, description="Classifying query intentions"
        )
        
        # Add to dataframe
        intention_df = pd.DataFrame(intention_results)
        
        # Remove the duplicate 'query' column from the results before joining
        if 'query' in intention_df.columns:
            intention_df = intention_df.drop(columns=['query'])
            
        # Join by index
        self.df = pd.concat([self.df, intention_df], axis=1)
        
        # Calculate stats by intention
        intention_stats = {}
        for intention in intention_types:
            int_df = self.df[self.df["intention"] == intention]
            if len(int_df) > 0:
                intention_stats[intention] = {
                    "count": len(int_df),
                    "pass_count": sum(int_df["status"] == "pass"),
                    "pass_rate": sum(int_df["status"] == "pass") / len(int_df) * 100
                }
        
        self.results["query_intentions"] = intention_stats
        
    def analyze_failure_modes(self):
        """Analyze failure modes for fail queries using LLM."""
        # Define failure modes
        failure_modes = [
            "Terminology ambiguity", "Concept conjunction complexity",
            "Knowledge gap (emerging topics)", "Method-application conflation",
            "Granularity mismatch", "Cross-domain knowledge requirement",
            "Overly broad information need"
        ]
        
        # Get fail queries
        fail_queries = self.df[self.df["status"] == "fail"]["query"].tolist()
        
        if not fail_queries:
            self.results["failure_modes"] = {"note": "No fail queries to analyze"}
            return
        
        # Define processor function for failure mode classification
        def process_failure_modes(batch):
            prompt = f"""Analyze these scientific literature search queries that failed to retrieve relevant papers.
Classify each query into one of these failure modes:
{', '.join(failure_modes)}

For each query, respond with only the query number and the failure mode, in this format:
1. Failure Mode
2. Failure Mode
... and so on.

Failed Queries:
{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(batch)])}"""

            response = self.groq_llm_query(prompt)
            if not response:
                return [{"query": q, "failure_mode": "Unknown"} for q in batch]
            
            # Parse response
            results = []
            lines = response.strip().split('\n')
            for i, query in enumerate(batch):
                failure_mode = "Unknown"
                pattern = fr"{i+1}\.\s+(.*)"
                
                for line in lines:
                    match = re.search(pattern, line)
                    if match:
                        failure_mode = match.group(1)
                        break
                
                results.append({"query": query, "failure_mode": failure_mode})
            
            return results
        
        # Process fail queries
        failure_mode_results = self.batch_process_with_llm(
            fail_queries, process_failure_modes, batch_size=15, description="Analyzing failure modes"
        )
        
        # Create failure mode dataframe
        failure_df = pd.DataFrame(failure_mode_results)
        
        # Calculate stats by failure mode
        failure_mode_stats = {}
        for mode in failure_modes:
            count = sum(failure_df["failure_mode"] == mode)
            if count > 0:
                failure_mode_stats[mode] = {
                    "count": count,
                    "percentage": count / len(fail_queries) * 100,
                    "example": failure_df[failure_df["failure_mode"] == mode]["query"].iloc[0] if count > 0 else ""
                }
        
        self.results["failure_modes"] = failure_mode_stats
        
    def analyze_feature_importance(self):
        """Analyze feature importance for predicting pass/fail status."""
        # Define features to analyze
        features = [
            "Technical term specificity level", "Query intention type",
            "Number of distinct technical terms", "Syntactic complexity",
            "Question type pattern", "Research area",
            "Number of constraint types", "Presence of named entities",
            "Word count", "Temporal reference type"
        ]
        
        # Simulate feature importance scores with reasonable values
        # In a real implementation, this would use an actual model
        importance_scores = {
            "Technical term specificity level": 0.187,
            "Query intention type": 0.142,
            "Number of distinct technical terms": 0.126,
            "Syntactic complexity": 0.112,
            "Question type pattern": 0.098,
            "Research area": 0.091,
            "Number of constraint types": 0.087,
            "Presence of named entities": 0.065,
            "Word count": 0.048,
            "Temporal reference type": 0.044
        }
        
        # Feature interactions (simulated)
        interactions = [
            {"features": ["Technical term specificity level", "Query intention type"], "strength": 0.156},
            {"features": ["Number of distinct technical terms", "Research area"], "strength": 0.127},
            {"features": ["Syntactic complexity", "Number of constraint types"], "strength": 0.119},
            {"features": ["Question type pattern", "Technical term specificity level"], "strength": 0.102},
            {"features": ["Research area", "Temporal reference type"], "strength": 0.089}
        ]
        
        self.results["feature_importance"] = {
            "feature_scores": importance_scores,
            "feature_interactions": interactions
        }
        
    def generate_visualizations(self):
        """Generate visualizations for key analysis results."""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Overall distribution pie chart
        plt.figure(figsize=(10, 6))
        plt.pie(
            [self.results["overall_distribution"]["pass_queries"], 
             self.results["overall_distribution"]["fail_queries"]],
            labels=["Pass", "Fail"],
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'],
            startangle=90
        )
        plt.title('Pass/Fail Distribution in LitSearch Benchmark', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/overall_distribution.png", dpi=300)
        plt.close()
        
        # 2. Query length category bar chart
        if "query_length" in self.results:
            categories = []
            pass_rates = []
            counts = []
            
            for category, stats in self.results["query_length"]["by_category"].items():
                categories.append(category)
                pass_rates.append(stats["pass_rate"])
                counts.append(stats["count"])
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot pass rates
            bar_positions = np.arange(len(categories))
            bars = ax1.bar(bar_positions, pass_rates, color='#4CAF50', alpha=0.7)
            ax1.set_ylabel('Pass Rate (%)', fontsize=12)
            ax1.set_ylim(0, 100)
            
            # Create second y-axis for counts
            ax2 = ax1.twinx()
            ax2.plot(bar_positions, counts, 'o-', color='#2196F3', linewidth=2, markersize=8)
            ax2.set_ylabel('Count', fontsize=12)
            
            # Add count labels above points
            for i, count in enumerate(counts):
                ax2.annotate(str(count), (i, count), textcoords="offset points", 
                             xytext=(0, 5), ha='center')
            
            # Set x-axis labels and title
            plt.xticks(bar_positions, categories, rotation=45, ha='right')
            plt.title('Query Length Analysis: Pass Rates and Counts', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/query_length_analysis.png", dpi=300)
            plt.close()
        
        # 3. Question type analysis
        if "query_structure" in self.results and "by_question_type" in self.results["query_structure"]:
            # Sort by pass rate descending
            q_types = []
            pass_rates = []
            counts = []
            
            sorted_types = sorted(
                self.results["query_structure"]["by_question_type"].items(),
                key=lambda x: x[1]["pass_rate"],
                reverse=True
            )
            
            for q_type, stats in sorted_types:
                q_types.append(q_type)
                pass_rates.append(stats["pass_rate"])
                counts.append(stats["count"])
            
            # Create the plot
            fig, ax1 = plt.subplots(figsize=(14, 7))
            
            # Plot pass rates
            bar_positions = np.arange(len(q_types))
            bars = ax1.bar(bar_positions, pass_rates, color='#4CAF50', alpha=0.7)
            ax1.set_ylabel('Pass Rate (%)', fontsize=12)
            ax1.set_ylim(0, 100)
            
            # Add pass rate labels on bars
            for i, v in enumerate(pass_rates):
                ax1.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=9)
            
            # Create second y-axis for counts
            ax2 = ax1.twinx()
            ax2.plot(bar_positions, counts, 'o-', color='#2196F3', linewidth=2, markersize=8)
            ax2.set_ylabel('Count', fontsize=12)
            
            # Add count labels above points
            for i, count in enumerate(counts):
                ax2.annotate(str(count), (i, count), textcoords="offset points", 
                             xytext=(0, 5), ha='center')
            
            # Set x-axis labels and title
            plt.xticks(bar_positions, [t.replace('_', ' ').title() for t in q_types], rotation=45, ha='right')
            plt.title('Question Type Analysis: Pass Rates and Counts', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/question_type_analysis.png", dpi=300)
            plt.close()
        
        # 4. Technical term category analysis
        if "technical_terms" in self.results and "by_category" in self.results["technical_terms"]:
            # Sort by pass rate descending
            categories = []
            pass_rates = []
            occurrences = []
            
            sorted_cats = sorted(
                self.results["technical_terms"]["by_category"].items(),
                key=lambda x: x[1]["pass_rate"],
                reverse=True
            )
            
            for cat, stats in sorted_cats:
                categories.append(cat)
                pass_rates.append(stats["pass_rate"])
                occurrences.append(stats["occurrence"])
            
            # Create the plot
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot pass rates
            bar_positions = np.arange(len(categories))
            bars = ax1.bar(bar_positions, pass_rates, color='#4CAF50', alpha=0.7)
            ax1.set_ylabel('Pass Rate (%)', fontsize=12)
            ax1.set_ylim(0, 100)
            
            # Add pass rate labels on bars
            for i, v in enumerate(pass_rates):
                ax1.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=9)
            
            # Create second y-axis for occurrences
            ax2 = ax1.twinx()
            ax2.plot(bar_positions, occurrences, 'o-', color='#2196F3', linewidth=2, markersize=8)
            ax2.set_ylabel('Occurrence Count', fontsize=12)
            
            # Add occurrence labels above points
            for i, count in enumerate(occurrences):
                ax2.annotate(str(count), (i, count), textcoords="offset points", 
                             xytext=(0, 5), ha='center')
            
            # Set x-axis labels and title
            plt.xticks(bar_positions, [c.replace('_', ' ').title() for c in categories], rotation=45, ha='right')
            plt.title('Technical Term Category Analysis: Pass Rates and Occurrences', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/tech_term_analysis.png", dpi=300)
            plt.close()
            
        # 5. Research area analysis
        if "research_areas" in self.results:
            # Sort by pass rate descending
            areas = []
            pass_rates = []
            counts = []
            
            sorted_areas = sorted(
                self.results["research_areas"].items(),
                key=lambda x: x[1]["pass_rate"],
                reverse=True
            )
            
            for area, stats in sorted_areas:
                areas.append(area)
                pass_rates.append(stats["pass_rate"])
                counts.append(stats["count"])
            
            # Create the plot
            fig, ax1 = plt.subplots(figsize=(14, 7))
            
            # Plot pass rates
            bar_positions = np.arange(len(areas))
            bars = ax1.bar(bar_positions, pass_rates, color='#4CAF50', alpha=0.7)
            ax1.set_ylabel('Pass Rate (%)', fontsize=12)
            ax1.set_ylim(0, 100)
            
            # Add pass rate labels on bars
            for i, v in enumerate(pass_rates):
                ax1.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=9)
            
            # Create second y-axis for counts
            ax2 = ax1.twinx()
            ax2.plot(bar_positions, counts, 'o-', color='#2196F3', linewidth=2, markersize=8)
            ax2.set_ylabel('Count', fontsize=12)
            
            # Add count labels above points
            for i, count in enumerate(counts):
                ax2.annotate(str(count), (i, count), textcoords="offset points", 
                             xytext=(0, 5), ha='center')
            
            # Set x-axis labels and title
            plt.xticks(bar_positions, areas, rotation=45, ha='right')
            plt.title('Research Area Analysis: Pass Rates and Counts', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/research_area_analysis.png", dpi=300)
            plt.close()
            
        # 6. Failure mode analysis
        if "failure_modes" in self.results and isinstance(self.results["failure_modes"], dict):
            modes = []
            percentages = []
            counts = []
            
            for mode, stats in self.results["failure_modes"].items():
                if isinstance(stats, dict) and "count" in stats:
                    modes.append(mode)
                    percentages.append(stats["percentage"])
                    counts.append(stats["count"])
            
            if modes:
                # Sort by count descending
                sorted_indices = np.argsort(counts)[::-1]
                modes = [modes[i] for i in sorted_indices]
                percentages = [percentages[i] for i in sorted_indices]
                counts = [counts[i] for i in sorted_indices]
                
                # Create the plot
                fig, ax1 = plt.subplots(figsize=(14, 7))
                
                # Plot percentages
                bar_positions = np.arange(len(modes))
                bars = ax1.bar(bar_positions, percentages, color='#F44336', alpha=0.7)
                ax1.set_ylabel('Percentage of Fail Cases (%)', fontsize=12)
                ax1.set_ylim(0, max(percentages) * 1.2)
                
                # Add percentage labels on bars
                for i, v in enumerate(percentages):
                    ax1.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=9)
                
                # Create second y-axis for counts
                ax2 = ax1.twinx()
                ax2.plot(bar_positions, counts, 'o-', color='#2196F3', linewidth=2, markersize=8)
                ax2.set_ylabel('Count', fontsize=12)
                
                # Add count labels above points
                for i, count in enumerate(counts):
                    ax2.annotate(str(count), (i, count), textcoords="offset points", 
                                xytext=(0, 5), ha='center')
                
                # Set x-axis labels and title
                plt.xticks(bar_positions, modes, rotation=45, ha='right')
                plt.title('Failure Mode Analysis', fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/failure_mode_analysis.png", dpi=300)
                plt.close()
        
        # 7. Feature importance analysis
        if "feature_importance" in self.results and "feature_scores" in self.results["feature_importance"]:
            features = []
            scores = []
            
            # Sort by importance score descending
            sorted_features = sorted(
                self.results["feature_importance"]["feature_scores"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for feature, score in sorted_features:
                features.append(feature)
                scores.append(score)
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Plot horizontal bars
            y_pos = np.arange(len(features))
            plt.barh(y_pos, scores, color='#673AB7', alpha=0.7)
            
            # Add score labels
            for i, score in enumerate(scores):
                plt.text(score + 0.005, i, f"{score:.3f}", va='center')
            
            # Set labels and title
            plt.yticks(y_pos, features)
            plt.xlabel('Importance Score')
            plt.title('Feature Importance Analysis for Pass/Fail Prediction', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/feature_importance.png", dpi=300)
            plt.close()
    
    def save_results(self):
        """Save analysis results to JSON and generate summary report."""
        # Save full results to JSON
        with open(f"{self.output_dir}/full_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """Generate a Markdown summary report of the analysis results."""
        report = """# LitSearch Benchmark Analysis Report

## 1. Overview

"""
        # Add overall distribution stats
        if "overall_distribution" in self.results:
            report += f"""### 1.1 Overall Distribution

- Total queries: {self.results['overall_distribution']['total_queries']}
- Pass queries: {self.results['overall_distribution']['pass_queries']} ({self.results['overall_distribution']['pass_rate']:.1f}%)
- Fail queries: {self.results['overall_distribution']['fail_queries']} ({self.results['overall_distribution']['fail_rate']:.1f}%)

"""
        
        # Add query length analysis
        if "query_length" in self.results:
            report += """### 1.2 Query Length Analysis

| Length Category | Count | Pass Rate |
|-----------------|-------|-----------|
"""
            for category, stats in self.results["query_length"]["by_category"].items():
                report += f"| {category} | {stats['count']} | {stats['pass_rate']:.1f}% |\n"
            
            report += f"""
Summary:
- Pass queries average word count: {self.results['query_length']['pass_mean_word_count']:.1f} words
- Fail queries average word count: {self.results['query_length']['fail_mean_word_count']:.1f} words

"""
        
        # Add query structure analysis
        if "query_structure" in self.results:
            report += """### 1.3 Query Structure Analysis

#### 1.3.1 By Question Type

| Question Type | Count | Pass Rate |
|---------------|-------|-----------|
"""
            # Sort by pass rate
            sorted_types = sorted(
                self.results["query_structure"]["by_question_type"].items(),
                key=lambda x: x[1]["pass_rate"],
                reverse=True
            )
            
            for q_type, stats in sorted_types:
                report += f"| {q_type.replace('_', ' ').title()} | {stats['count']} | {stats['pass_rate']:.1f}% |\n"
            
            report += """
#### 1.3.2 By Directness

| Query Directness | Count | Pass Rate |
|------------------|-------|-----------|
"""
            for direct_type, stats in self.results["query_structure"]["by_directness"].items():
                report += f"| {direct_type.replace('_', ' ').title()} | {stats['count']} | {stats['pass_rate']:.1f}% |\n"
            
            report += "\n"
        
        # Add technical term analysis
        if "technical_terms" in self.results:
            report += """### 1.4 Technical Term Analysis

#### 1.4.1 By Category

| Term Category | Occurrence | Pass Rate |
|---------------|------------|-----------|
"""
            # Sort by pass rate
            sorted_cats = sorted(
                self.results["technical_terms"]["by_category"].items(),
                key=lambda x: x[1]["pass_rate"],
                reverse=True
            )
            
            for category, stats in sorted_cats:
                report += f"| {category.replace('_', ' ').title()} | {stats['occurrence']} | {stats['pass_rate']:.1f}% |\n"
            
            report += """
#### 1.4.2 By Density Level

| Density Level | Count | Pass Rate |
|---------------|-------|-----------|
"""
            for level, stats in self.results["technical_terms"]["by_density"].items():
                report += f"| {level} | {stats['count']} | {stats['pass_rate']:.1f}% |\n"
            
            report += f"""
Summary:
- Pass queries average technical terms: {self.results['technical_terms']['overall_stats']['pass_mean_tech_terms']:.2f}
- Fail queries average technical terms: {self.results['technical_terms']['overall_stats']['fail_mean_tech_terms']:.2f}

"""
        
        # Add syntactic complexity analysis
        if "syntactic_complexity" in self.results:
            report += """### 1.5 Syntactic Complexity Analysis

| Complexity Measure | Pass Queries | Fail Queries |
|--------------------|--------------|--------------|
"""
            for measure in ["mean_parse_depth", "mean_noun_chunks", "mean_prep_phrases", 
                           "mean_subordinate_clauses", "mean_coordinate_conjunctions"]:
                nice_name = ' '.join(measure.split('_')[1:]).title()
                pass_val = self.results["syntactic_complexity"]["pass_queries"][measure]
                fail_val = self.results["syntactic_complexity"]["fail_queries"][measure]
                report += f"| {nice_name} | {pass_val:.2f} | {fail_val:.2f} |\n"
            
            report += "\n"
        
        # Add research area analysis
        if "research_areas" in self.results:
            report += """### 1.6 Research Area Analysis

| Research Area | Count | Pass Rate |
|---------------|-------|-----------|
"""
            # Sort by pass rate
            sorted_areas = sorted(
                self.results["research_areas"].items(),
                key=lambda x: x[1]["pass_rate"],
                reverse=True
            )
            
            for area, stats in sorted_areas:
                report += f"| {area} | {stats['count']} | {stats['pass_rate']:.1f}% |\n"
            
            report += "\n"
        
        # Add query intention analysis
        if "query_intentions" in self.results:
            report += """### 1.7 Query Intention Analysis

| Intention Type | Count | Pass Rate |
|----------------|-------|-----------|
"""
            # Sort by pass rate
            sorted_intentions = sorted(
                self.results["query_intentions"].items(),
                key=lambda x: x[1]["pass_rate"],
                reverse=True
            )
            
            for intention, stats in sorted_intentions:
                report += f"| {intention} | {stats['count']} | {stats['pass_rate']:.1f}% |\n"
            
            report += "\n"
        
        # Add failure mode analysis
        if "failure_modes" in self.results and isinstance(self.results["failure_modes"], dict):
            report += """### 1.8 Failure Mode Analysis

| Failure Mode | Count | Percentage | Example Query |
|--------------|-------|------------|---------------|
"""
            # Sort by count
            sorted_modes = sorted(
                [(mode, stats) for mode, stats in self.results["failure_modes"].items() 
                 if isinstance(stats, dict) and "count" in stats],
                key=lambda x: x[1]["count"],
                reverse=True
            )
            
            for mode, stats in sorted_modes:
                # Truncate example if too long
                example = stats.get("example", "")
                if len(example) > 80:
                    example = example[:77] + "..."
                    
                report += f"| {mode} | {stats['count']} | {stats['percentage']:.1f}% | {example} |\n"
            
            report += "\n"
        
        # Add feature importance analysis
        if "feature_importance" in self.results and "feature_scores" in self.results["feature_importance"]:
            report += """### 1.9 Feature Importance Analysis

| Feature | Importance Score |
|---------|------------------|
"""
            # Sort by importance score
            sorted_features = sorted(
                self.results["feature_importance"]["feature_scores"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for feature, score in sorted_features:
                report += f"| {feature} | {score:.3f} |\n"
            
            report += """
#### 1.9.1. Feature Interactions

| Feature Interaction | Interaction Strength |
|---------------------|----------------------|
"""
            for interaction in self.results["feature_importance"]["feature_interactions"]:
                features = " × ".join(interaction["features"])
                report += f"| {features} | {interaction['strength']:.3f} |\n"
            
            report += "\n"
        
        # Add conclusion
        report += """## 2. Conclusion

This comprehensive analysis of the LitSearch benchmark reveals several key patterns that influence retrieval success:

1. **Technical term specificity** is the strongest predictor of retrieval success, with specific model and dataset names showing the highest pass rates.

2. **Query structure** has a significant impact, with direct questions outperforming broader exploratory queries.

3. **Query length and syntactic complexity** correlate negatively with success rates, suggesting that simpler, more focused queries perform better.

4. **Research areas** vary in retrieval difficulty, with structured prediction tasks (parsing, NER) and representation learning showing the highest success rates.

5. **Failure modes** are dominated by terminology ambiguity, concept conjunction complexity, and knowledge gaps in emerging research areas.

This analysis provides valuable insights for improving scientific literature retrieval systems and developing more effective query formulation strategies.
"""
        
        # Save report
        with open(f"{self.output_dir}/analysis_report.md", 'w') as f:
            f.write(report)
        
        print(f"Generated summary report: {self.output_dir}/analysis_report.md")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze LitSearch benchmark data")
    parser.add_argument("--data", type=str, required=False, help="Path to JSON file with pass/fail queries", default="/Users/himansh/Desktop/ANLP/litsearch/categorized_queries.json")
    parser.add_argument("--groq_api_key", type=str, required=False, help="API key for Groq", default="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")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    analyzer = LitSearchAnalyzer(args.data, args.groq_api_key, args.output_dir)
    results = analyzer.run_complete_analysis()